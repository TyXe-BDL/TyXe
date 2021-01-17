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
    return (x,)


def _to(x, device):
    return map(lambda t: t.to(device), _as_tuple(x))


class _BNN(pynn.PyroModule):

    def __init__(self, net, prior, name=""):
        super().__init__(name)
        self.net = net
        pynn.module.to_pyro_module_(self.net)
        self.prior = prior
        self.prior.apply_(self.net)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def update_prior(self, new_prior):
        self.prior = new_prior
        self.prior.update_(self.net)


class GuidedBNN(_BNN):

    def __init__(self, net, prior, guide_builder=None, name=""):
        super().__init__(net, prior, name=name)
        self.net_guide = guide_builder(self.net) if guide_builder is not None else _empty_guide

    def guided_forward(self, *args, guide_tr=None, **kwargs):
        if guide_tr is None:
            guide_tr = poutine.trace(self.net_guide).get_trace(*args, **kwargs)
        return poutine.replay(self.net, trace=guide_tr)(*args, **kwargs)


class PytorchBNN(GuidedBNN):

    def __init__(self, net, prior, guide_builder=None, name="", closed_form_kl=True):
        super().__init__(net, prior, guide_builder=guide_builder, name=name)
        self.cached_output = None
        self.cached_kl_loss = None
        self._loss = TraceMeanField_ELBO() if closed_form_kl else Trace_ELBO()

    def named_pytorch_parameters(self, *input_data):
        model_trace = poutine.trace(self.net, param_only=True).get_trace(*input_data)
        guide_trace = poutine.trace(self.net_guide, param_only=True).get_trace(*input_data)
        for name, msg in itertools.chain(model_trace.nodes.items(), guide_trace.nodes.items()):
            yield name, msg["value"].unconstrained()
        # yield from poutine.trace(self, param_only=True).get_trace(*input_data).nodes.items()

    def pytorch_parameters(self, input_data_or_fwd_fn):
        yield from map(itemgetter(1), self.named_pytorch_parameters(input_data_or_fwd_fn))

    def cached_forward(self, *args, **kwargs):
        self.cached_output = super().forward(*args, **kwargs)
        return self.cached_output

    def forward(self, *args, **kwargs):
        self.cached_kl_loss = self._loss.differentiable_loss(self.cached_forward, self.net_guide, *args, **kwargs)
        return self.cached_output


class VariationalBNN(GuidedBNN):

    def __init__(self, net, prior, observation_model, net_guide_builder=None, observation_guide_builder=None, name=""):
        super().__init__(net, prior, net_guide_builder, name=name)
        self.observation_model = observation_model
        weight_sample_sites = list(util.pyro_sample_sites(self.net))
        if observation_guide_builder is not None:
            self.observation_guide = observation_guide_builder(poutine.block(
                self.model, hide=weight_sample_sites + [self.observation_model.observation_name]))
        else:
            self.observation_guide = _empty_guide

    def model(self, x, obs=None):
        predictions = self(*_as_tuple(x))
        self.observation_model(predictions, obs)
        return predictions

    def guide(self, x, obs=None):
        result = self.net_guide(*_as_tuple(x)) or {}
        result.update(self.observation_guide(*_as_tuple(x), obs) or {})
        return result

    def fit(self, data_loader, optim, num_epochs, callback=None, num_particles=1, closed_form_kl=True, device=None):
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
        return self.observation_model.aggregate_predictions(predictions) if aggregate else predictions

    def evaluate(self, input_data, y, num_predictions=1, aggregate=True, reduction="sum"):
        predictions = self.predict(*_as_tuple(input_data), num_predictions=num_predictions, aggregate=aggregate)
        error = self.observation_model.error(predictions, y, reduction=reduction)
        ll = self.observation_model.log_likelihood(predictions, y, reduction=reduction)
        return error, ll


class MCMC_BNN(_BNN):

    def __init__(self, net, prior, observation_model, kernel_builder, name=""):
        super().__init__(net, prior, name=name)
        self.observation_model = observation_model
        self.kernel = kernel_builder(self.model)
        self._mcmc = None

    def model(self, x, obs=None):
        predictions = self(x)
        self.observation_model(predictions, obs)
        return predictions

    def fit(self, input_data, observation_data, num_samples, device=None, **mcmc_kwargs):
        # data = map(lambda t: torch.cat(t).to(device), zip(*iter(data_loader)))
        self._mcmc = MCMC(self.kernel, num_samples, **mcmc_kwargs)
        self._mcmc.run(input_data, observation_data)

        return self._mcmc

    def predict(self, x, num_predictions=1, aggregate=True):
        if self._mcmc is None:
            raise RuntimeError("Call .fit to run MCMC and obtain samples from the posterior first.")

        preds = []
        weight_samples = self._mcmc.get_samples(num_samples=num_predictions)
        with torch.no_grad():
            for i in range(num_predictions):
                weights = {name: sample[i] for name, sample in weight_samples.items()}
                preds.append(poutine.condition(self, weights)(x))
        predictions = torch.stack(preds)
        return self.observation_model.aggregate_predictions(predictions) if aggregate else predictions
