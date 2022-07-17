from collections import defaultdict
import itertools
from operator import itemgetter

import torch

import pyro.nn as pynn
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, MCMC

from . import util


__all__ = ["PytorchBNN", "VariationalBNN", "MCMC_BNN"]


def _empty_guide(*args, **kwargs):
    return {}


def _as_tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return x,


def _to(x, device):
    return map(lambda t: t.to(device) if device is not None else t, _as_tuple(x))


class _BNN(pynn.PyroModule):
    """BNN base class that takes an nn.Module, turns it into a PyroModule and applies a prior to it, i.e. replaces
    nn.Parameter attributes by PyroSamples according to the specification in the prior. The forward method wraps the
    forward pass of the net and samples weights from the prior distributions.

    :param nn.Module net: pytorch neural network to be turned into a BNN.
    :param prior tyxe.priors.Prior: prior object that specifies over which parameters we want uncertainty.
    :param str name: base name for the BNN PyroModule."""

    def __init__(self, net, prior, name=""):
        super().__init__(name)
        self.net = net
        pynn.module.to_pyro_module_(self.net)
        self.prior = prior
        self.prior.apply_(self.net)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def update_prior(self, new_prior):
        """Uppdates the prior of the network, i.e. calls its update_ method on the net.

        :param tyxe.priors.Prior new_prior: Prior for replacing the previous prior, i.e. substituting the PyroSample
            attributes of the net."""
        self.prior = new_prior
        self.prior.update_(self)


class GuidedBNN(_BNN):
    """Guided BNN class that in addition to the network and prior also has a guide for doing approximate inference
    over the neural network weights. The guide_builder argument is called on the net after it has been transformed to
    a PyroModule and returns the pyro guide function that sample from the approximate posterior.

    :param callable guide_builder: callable that takes a probabilistic pyro function with sample statements and returns
        an object that helps with inference, i.e. a callable guide function that samples from an approximate posterior
        for variational BNNs or an MCMC kernel for MCMC-based BNNs. May be None for maximum likelihood inference if
        the prior leaves all parameters of the net as such."""

    def __init__(self, net, prior, guide_builder=None, name=""):
        super().__init__(net, prior, name=name)
        self.net_guide = guide_builder(self.net) if guide_builder is not None else _empty_guide

    def guided_forward(self, *args, guide_tr=None, **kwargs):
        if guide_tr is None:
            guide_tr = poutine.trace(self.net_guide).get_trace(*args, **kwargs)
        return poutine.replay(self.net, trace=guide_tr)(*args, **kwargs)


class PytorchBNN(GuidedBNN):
    """Low-level variational BNN class that can serve as a drop-in replacement for an nn.Module.

    :param bool closed_form_kl: whether to use TraceMeanField_ELBO or Trace_ELBO as the loss, i.e. calculate KL
        divergences in closed form or via a Monte Carlo approximate of the difference of log densities between
        variational posterior and prior."""

    def __init__(self, net, prior, guide_builder=None, name="", closed_form_kl=True):
        super().__init__(net, prior, guide_builder=guide_builder, name=name)
        self.cached_output = None
        self.cached_kl_loss = None
        self._loss = TraceMeanField_ELBO() if closed_form_kl else Trace_ELBO()

    def named_pytorch_parameters(self, *input_data):
        """Equivalent of the named_parameters method of an nn.Module. Ensures that prior and guide are run once to
        initialize all pyro parameters. Those are then collected and returned via the trace poutine."""
        model_trace = poutine.trace(self.net, param_only=True).get_trace(*input_data)
        guide_trace = poutine.trace(self.net_guide, param_only=True).get_trace(*input_data)
        for name, msg in itertools.chain(model_trace.nodes.items(), guide_trace.nodes.items()):
            yield name, msg["value"].unconstrained()

    def pytorch_parameters(self, input_data_or_fwd_fn):
        yield from map(itemgetter(1), self.named_pytorch_parameters(input_data_or_fwd_fn))

    def cached_forward(self, *args, **kwargs):
        # cache the output of forward to make it effectful, so that we can access the output when running forward with
        # posterior rather than prior samples
        self.cached_output = super().forward(*args, **kwargs)
        return self.cached_output

    def forward(self, *args, **kwargs):
        self.cached_kl_loss = self._loss.differentiable_loss(self.cached_forward, self.net_guide, *args, **kwargs)
        return self.cached_output


class _SupervisedBNN(GuidedBNN):
    """Base class for supervised BNNs that defines the interface of the predict method and implements
    evaluate. Agnostic to the kind of inference performed.

    :param tyxe.likelihoods.Likelihood likelihood: Likelihood object that implements a forward method including
        a pyro.sample statement for labelled data given neural network predictions and implements logic for aggregating
        multiple predictions and evaluating them."""

    def __init__(self, net, prior, likelihood, net_guide_builder=None, name=""):
        super().__init__(net, prior, net_guide_builder, name=name)
        self.likelihood = likelihood

    def model(self, x, obs=None):
        predictions = self(*_as_tuple(x))
        self.likelihood(predictions, obs)
        return predictions

    def evaluate(self, input_data, y, num_predictions=1, aggregate=True, reduction="sum"):
        """"Utility method for evaluation. Calculates a likelihood-dependent errors measure, e.g. squared errors or
        mis-classifications and

        :param input_data: Inputs to the neural net. Must be a tuple of more than one.
        :param y: observations, e.g. class labels.
        :param int num_predictions: number of forward passes.
        :param bool aggregate: whether to aggregate the outputs of the forward passes before evaluating.
        :param str reduction: "sum", "mean" or "none". How to process the tensor of errors. "sum" adds them up,
            "mean" averages them and "none" simply returns the tensor."""
        predictions = self.predict(*_as_tuple(input_data), num_predictions=num_predictions, aggregate=aggregate)
        error = self.likelihood.error(predictions, y, reduction=reduction)
        ll = self.likelihood.log_likelihood(predictions, y, reduction=reduction)
        return error, ll

    def predict(self, *input_data, num_predictions=1, aggregate=True):
        """Makes predictions on the input data

        :param input_data: inputs to the neural net, e.g. torch.Tensors
        :param int num_predictions: number of forward passes through the net
        :param bool aggregate: whether to aggregate the predictions depending on the likelihood, e.g. averaging them."""
        raise NotImplementedError


class VariationalBNN(_SupervisedBNN):
    """Variational BNN class for supervised problems. Requires a likelihood that describes the data noise and an
    optional guide builder for it should it contain any variables that need to be inferred. Provides high-level utility
    method such as fit, predict and

    :param callable net_guide_builder: pyro.infer.autoguide.AutoCallable style class that given a pyro function
        constructs a variational posterior that sample the same unobserved sites from distributions with learnable
        parameters.
    :param callable likelihood_guide_builder: optional callable that constructs a guide for the likelihood if it
        contains any unknown variable, such as the precision/scale of a Gaussian."""
    def __init__(self, net, prior, likelihood, net_guide_builder=None, likelihood_guide_builder=None, name=""):
        super().__init__(net, prior, likelihood, net_guide_builder, name=name)
        weight_sample_sites = list(util.pyro_sample_sites(self.net))
        if likelihood_guide_builder is not None:
            self.likelihood_guide = likelihood_guide_builder(poutine.block(
                self.model, hide=weight_sample_sites + [self.likelihood.data_name]))
        else:
            self.likelihood_guide = _empty_guide

    def guide(self, x, obs=None):
        result = self.net_guide(*_as_tuple(x)) or {}
        result.update(self.likelihood_guide(*_as_tuple(x), obs) or {})
        return result

    def fit(self, data_loader, optim, num_epochs, callback=None, num_particles=1, closed_form_kl=True, device=None):
        """Optimizes the variational parameters on data from data_loader using optim for num_epochs.

        :param Iterable data_loader: iterable over batches of data, e.g. a torch.utils.data.DataLoader. Assumes that
            each element consists of a length two tuple of list, with the first element either containing a single
            object or a list of objects, e.g. torch.Tensors, that are the inputs to the neural network. The second
            element is a single torch.Tensor e.g. of class labels.
        :param optim: pyro optimizer to be used for constructing an SVI object, e.g. pyro.optim.Adam({"lr": 1e-3}).
        :param int num_epochs: number of passes over data_loader.
        :param callable callback: optional function to invoke after every training epoch. Receives the BNN object,
            the epoch number and the average value of the ELBO over the epoch. May return True to terminate
            optimization before num_epochs, e.g. if it finds that a validation log likelihood saturates.
        :param int num_particles: number of MC samples for estimating the ELBO.
        :param bool closed_form_kl: whether to use TraceMeanField_ELBO or Trace_ELBO, i.e. calculate KL divergence
            between approximate posterior and prior in closed form or via a Monte Carlo estimate.
        :param torch.device device: optional device to send the data to.
        """
        old_training_state = self.net.training
        self.net.train(True)

        loss = TraceMeanField_ELBO(num_particles) if closed_form_kl else Trace_ELBO(num_particles)
        svi = SVI(self.model, self.guide, optim, loss=loss)

        for i in range(num_epochs):
            elbo = 0.
            num_batch = 1
            for num_batch, (input_data, observation_data) in enumerate(iter(data_loader), 1):
                elbo += svi.step(tuple(_to(input_data, device)), tuple(_to(observation_data, device))[0])

            # the callback can stop training by returning True
            if callback is not None and callback(self, i, elbo / num_batch):
                break

        self.net.train(old_training_state)
        return svi

    def predict(self, *input_data, num_predictions=1, aggregate=True, guide_traces=None):
        if guide_traces is None:
            guide_traces = [None] * num_predictions

        preds = []
        with torch.autograd.no_grad():
            for trace in guide_traces:
                preds.append(self.guided_forward(*input_data, guide_tr=trace))
        predictions = torch.stack(preds)
        return self.likelihood.aggregate_predictions(predictions) if aggregate else predictions


# TODO inherit from _SupervisedBNN to unify the class hierarchy. This will require changing the GuidedBNN baseclass to
#  construct the guide on top of self.model rather than self.net (model of GuidedBNN could just call the net and
#  SupervisedBNN adds the likelihood on top) and consequently removing the likelihood_guide_builder parameter for
#  the VariationalBNN class. This will however require hiding the likelihood.data site from the guide_builder in the
#  base class.
class MCMC_BNN(_BNN):
    """Supervised BNN class with an interface to pyro's MCMC that is unified with the VariationalBNN class.

    :param callable kernel_builder: function or class that returns an object that will accepted as kernel by
        pyro.infer.mcmc.MCMC, e.g. pyro.infer.mcmc.HMC or NUTS. Will be called with the entire model, i.e. also
        infer variables in the likelihood."""

    def __init__(self, net, prior, likelihood, kernel_builder, name=""):
        super().__init__(net, prior, name=name)
        self.likelihood = likelihood
        self.kernel = kernel_builder(self.model)
        self._mcmc = None

    def model(self, x, obs=None):
        predictions = self(*_as_tuple(x))
        self.likelihood(predictions, obs)
        return predictions

    def fit(self, data_loader, num_samples, device=None, batch_data=False, **mcmc_kwargs):
        """Runs MCMC on the data from data loader using the kernel that was used to instantiate the class.

        :param data_loader: iterable or list of batched inputs to the net. If iterable treated like the data_loader
            of VariationalBNN and all network inputs are concatenated via torch.cat. Otherwise must be a tuple of
            a single or list of network inputs and a tensor for the targets.
        :param int num_samples: number of MCMC samples to draw.
        :param torch.device device: optional device to send the data to.
        :param batch_data: whether to treat data_loader as a full batch of data or an iterable over mini-batches.
        :param dict mcmc_kwargs: keyword arguments for initializing the pyro.infer.mcmc.MCMC object."""
        if batch_data:
            input_data, observation_data = data_loader
        else:
            input_data_lists = defaultdict(list)
            observation_data_list = []
            for in_data, obs_data in iter(data_loader):
                for i, data in enumerate(_as_tuple(in_data)):
                    input_data_lists[i].append(data.to(device))
                observation_data_list.append(obs_data.to(device))
            input_data = tuple(torch.cat(input_data_lists[i]) for i in range(len(input_data_lists)))
            observation_data = torch.cat(observation_data_list)
        self._mcmc = MCMC(self.kernel, num_samples, **mcmc_kwargs)
        self._mcmc.run(input_data, observation_data)

        return self._mcmc

    def predict(self, *input_data, num_predictions=1, aggregate=True):
        if self._mcmc is None:
            raise RuntimeError("Call .fit to run MCMC and obtain samples from the posterior first.")

        preds = []
        weight_samples = self._mcmc.get_samples(num_samples=num_predictions)
        with torch.no_grad():
            for i in range(num_predictions):
                weights = {name: sample[i] for name, sample in weight_samples.items()}
                preds.append(poutine.condition(self, weights)(*input_data))
        predictions = torch.stack(preds)
        return self.likelihood.aggregate_predictions(predictions) if aggregate else predictions
