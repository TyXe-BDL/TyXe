import torch
import torch.nn.functional as F
import torch.distributions.utils as dist_utils
import torch.distributions as torchdist
from torch.distributions import transforms

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


def inverse_softplus(t):
    return t.expm1().log()


def _reduce(tensor, reduction):
    if reduction == "none":
        return tensor
    elif reduction == "sum":
        return tensor.sum()
    elif reduction == "mean":
        return tensor.mean()
    else:
        raise ValueError("Invalid reduction: '{}'. Must be one of ('none', 'sum', 'mean').".format(reduction))


def _make_name(prefix, suffix):
    return ".".join([prefix, suffix]) if prefix else suffix


class ObservationModel(PyroModule):

    def __init__(self, dataset_size, event_dim=0, name="", observation_name="obs"):
        super().__init__(name)
        self.dataset_size = dataset_size
        self.event_dim = event_dim
        self._obs_name = observation_name

    @property
    def observation_name(self):
        return self.var_name(self._obs_name)

    def var_name(self, name):
        return _make_name(self._pyro_name, name)

    def forward(self, predictions, obs=None):
        predictive_distribution = self.predictive_distribution(predictions)
        with pyro.plate(self.var_name("data"), subsample=predictions, size=self.dataset_size):
            return pyro.sample(self.observation_name, predictive_distribution, obs=obs)

    def log_likelihood(self, predictions, data, aggregation_dim=None, reduction="none"):
        if aggregation_dim is not None:
            predictions = self.aggregate_predictions(predictions, aggregation_dim)
        log_probs = self.predictive_distribution(predictions).log_prob(data)
        return _reduce(log_probs, reduction)

    def error(self, predictions, data, aggregation_dim=None, reduction="none"):
        if aggregation_dim is not None:
            predictions = self.aggregate_predictions(predictions, aggregation_dim)
        errors = self._calc_error(self._point_predictions(predictions), data)
        return _reduce(errors, reduction)

    def sample(self, predictions, sample_shape=torch.Size()):
        return self.predictive_distribution(predictions).sample(sample_shape)

    def predictive_distribution(self, predictions):
        return self.batch_predictive_distribution(predictions).to_event(self.event_dim)

    def batch_predictive_distribution(self, predictions):
        raise NotImplementedError

    def aggregate_predictions(self, predictions, dim=0):
        raise NotImplementedError

    def _point_predictions(self, predictions):
        raise NotImplementedError

    def _calc_error(self, point_predictions, data):
        raise NotImplementedError


# TODO add required binary class attribute to move _aggregate_predictions method from Bernoulli/Categorical to _Discrete
class _Discrete(ObservationModel):

    def __init__(self, dataset_size, logit_predictions=True, event_dim=0, name="", observation_name="obs"):
        super().__init__(dataset_size, event_dim=event_dim, name=name, observation_name=observation_name)
        self.logit_predictions = logit_predictions

    def base_dist(self, probs=None, logits=None):
        raise NotImplementedError

    def batch_predictive_distribution(self, predictions):
        return self.base_dist(logits=predictions) if self.logit_predictions else self.base_dist(probs=predictions)

    def _calc_error(self, point_predictions, data):
        return point_predictions.ne(data).float()


class Bernoulli(_Discrete):

    base_dist = dist.Bernoulli

    def _point_predictions(self, predictions):
        return predictions.gt(0.) if self.logit_predictions else predictions.gt(0.5)

    def aggregate_predictions(self, predictions, dim=0):
        probs = dist_utils.logits_to_probs(predictions, is_binary=True) if self.logit_predictions else predictions
        avg_probs = probs.mean(dim)
        return dist_utils.probs_to_logits(avg_probs, is_binary=True) if self.logit_predictions else avg_probs


class Categorical(_Discrete):

    base_dist = dist.Categorical

    def _point_predictions(self, predictions):
        return predictions.argmax(-1)

    def aggregate_predictions(self, predictions, dim=0):
        probs = dist_utils.logits_to_probs(predictions, is_binary=False) if self.logit_predictions else predictions
        avg_probs = probs.mean(dim)
        return dist_utils.probs_to_logits(avg_probs, is_binary=False) if self.logit_predictions else avg_probs


class Gaussian(ObservationModel):

    def __init__(self, dataset_size, event_dim=1, name="", observation_name="obs"):
        super().__init__(dataset_size, event_dim=event_dim, name=name, observation_name=observation_name)
        self.event_dim = event_dim

    def batch_predictive_distribution(self, predictions):
        loc, scale = self._predictive_loc_scale(predictions)
        return dist.Normal(loc, scale)

    def _point_predictions(self, predictions):
        return self._predictive_loc_scale(predictions)[0]

    def _calc_error(self, point_predictions, data):
        return point_predictions.sub(data).pow(2)

    def _predictive_loc_scale(self, predictions):
        raise NotImplementedError


class HeteroskedasticGaussian(Gaussian):

    def __init__(self, dataset_size, positive_scale=False, event_dim=1, name="", observation_name="obs"):
        super().__init__(dataset_size, event_dim=event_dim, name=name, observation_name=observation_name)
        self.positive_scale = positive_scale

    def aggregate_predictions(self, predictions, dim=0):
        loc, scale = self._predictive_loc_scale(predictions)
        precision = scale.pow(-2)
        total_precision = precision.sum(dim)
        agg_loc = loc.mul(precision).sum(dim).div(total_precision)
        agg_scale = precision.reciprocal().mean(dim).add(loc.var(dim)).sqrt()
        if not self.positive_scale:
            agg_scale = inverse_softplus(agg_scale)
        return torch.cat([agg_loc, agg_scale], -1)

    def _predictive_loc_scale(self, predictions):
        loc, pred_scale = predictions.chunk(2, dim=-1)
        scale = pred_scale if self.positive_scale else F.softplus(pred_scale)
        return loc, scale


class HomoskedasticGaussian(Gaussian):

    def __init__(self, dataset_size, scale=None, precision=None, event_dim=1, name="", observation_name="obs"):
        super().__init__(dataset_size, event_dim=event_dim, name=name, observation_name=observation_name)
        if int(scale is None) + int(precision is None) != 1:
            raise ValueError("Exactly one of scale and precision must be specified")
        elif isinstance(scale, (dist.Distribution, torchdist.Distribution)):
            # if the scale or precision is a distribution, that is used as the prior for a PyroSample. I'm not
            # completely sure if it is a good idea to allow regular pytorch distributions, since they might not have the
            # correct event_dim, so perhaps it's safer to check e.g. if the batch shape is empty and raise an error
            # otherwise
            precision = PyroSample(prior=dist.TransformedDistribution(scale, transforms.PowerTransform(-2.)))
            scale = PyroSample(prior=scale)
        elif isinstance(precision, (dist.Distribution, torchdist.Distribution)):
            scale = PyroSample(prior=dist.TransformedDistribution(precision, transforms.PowerTransform(-0.5)))
            precision = PyroSample(prior=precision)
        else:
            # nothing to do, precision or scale is a number/tensor/parameter
            pass
        self._scale = scale
        self._precision = precision

    @property
    def scale(self):
        if self._scale is None:
            return self.precision ** -0.5
        else:
            return self._scale

    @property
    def precision(self):
        if self._precision is None:
            return self.scale ** -2
        else:
            return self._precision

    def aggregate_predictions(self, predictions, dim=0):
        loc = predictions.mean(dim)
        scale = predictions.var(dim).add(self.scale ** 2).sqrt()
        return loc, scale

    def _predictive_loc_scale(self, predictions):
        if isinstance(predictions, tuple):
            loc, scale = predictions
        else:
            loc = predictions
            scale = self.scale
        return loc, scale
