import torch

import pyro.nn as pynn
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, MCMC


from . import util


def _empty_guide(*args, **kwargs):
    return {}


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


class SupervisedBNN(GuidedBNN):

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
        predictions = self(x)
        self.observation_model(predictions, obs)
        return predictions

    def guide(self, x, obs=None):
        result = self.net_guide(x) or {}
        result.update(self.observation_guide(x, obs) or {})
        return result

    def fit(self, data_loader, optim, num_epochs, callback=None, closed_form_kl=True, device=None):
        old_training_state = self.net.training
        self.net.train(True)

        loss = TraceMeanField_ELBO() if closed_form_kl else Trace_ELBO()
        svi = SVI(self.model, self.guide, optim, loss=loss)

        for i in range(num_epochs):
            elbo = 0.
            num_batch = 1
            for num_batch, data in enumerate(iter(data_loader), 1):
                elbo += svi.step(*map(lambda t: t.to(device), data))

            # the callback can stop training by returning True
            if callback is not None and callback(self, i, elbo / num_batch):
                break

        self.net.train(old_training_state)
        return svi

    def predict(self, x, num_predictions=1, aggregate=True, guide_traces=None):
        if guide_traces is None:
            guide_traces = [None] * num_predictions

        preds = []
        with torch.autograd.no_grad():
            for trace in guide_traces:
                preds.append(self.guided_forward(x, guide_tr=trace))
        predictions = torch.stack(preds)
        return self.observation_model.aggregate_predictions(predictions) if aggregate else predictions

    def evaluate(self, x, y, num_predictions=1, aggregate=True, reduction="sum"):
        predictions = self.predict(x, num_predictions, aggregate=aggregate)
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

    def fit(self, data_loader, num_samples, device=None, **mcmc_kwargs):
        data = map(lambda t: torch.cat(t).to(device), zip(*iter(data_loader)))
        self._mcmc = MCMC(self.kernel, num_samples, **mcmc_kwargs)
        self._mcmc.run(*data)

    def predict(self, x, num_predictions=1, aggregate=True):
        if self._mcmc is None:
            raise RuntimeError("Call .fit to run MCMC and obtain samples from the posterior first.")

        preds = []
        weight_samples = self._mcmc.get_samples(num_samples=num_predictions)
        with torch.no_grad():
            for i in range(num_predictions):
                weights = {name: sample[i] for name, sample in weight_samples.items()}
                preds.append(poutine.condition(self.model, weights)(x))
        predictions = torch.stack(preds)
        return self.observation_model.aggregate_predictions(predictions) if aggregate else predictions
