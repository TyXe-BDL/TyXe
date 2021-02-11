import torch
import torch.nn.functional as F
import torch.distributions.utils as dist_utils
import torch.distributions as torchdist
from torch.distributions import transforms

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


__all__ = ["Bernoulli", "Categorical", "HomoskedasticGaussian", "HeteroskedasticGaussian"]


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


class Likelihood(PyroModule):
    """Base class for BNN likelihoods. PyroModule wrapper around the most common distribution class for data noise.
    The forward method draws a pyro sample to be used in a model function given some predictions. log_likelihood and
    error are utility functions for evaluation.

    :param int dataset_size: Number of data points in the dataset for rescaling the log likelihood in the forward
        method when using mini-batches. May be None to disable rescaling.
    :param int event_dim: Number of dimensions of the predictive distribution to be interpreted as independent.
    :param str name: Base name of the PyroModule.
    :param str data_name: Site name of the pyro sample for the data in forward."""

    def __init__(self, dataset_size, event_dim=0, name="", data_name="data"):
        super().__init__(name)
        self.dataset_size = dataset_size
        self.event_dim = event_dim
        self._data_name = data_name

    @property
    def data_name(self):
        return self.var_name(self._data_name)

    def var_name(self, name):
        return _make_name(self._pyro_name, name)

    def forward(self, predictions, obs=None):
        """Executes a pyro sample statement to sample from the distribution corresponding to the likelihood class
        given some predictions. The values of the sample can set to some optional observations obs.

        :param torch.Tensor predictions: tensor of predictions.
        :param torch.Tensor obs: optional known values for the samples."""
        predictive_distribution = self.predictive_distribution(predictions)
        if predictive_distribution.batch_shape:
            dataset_size = self.dataset_size if self.dataset_size is not None else len(predictions)
            with pyro.plate(self.data_name+"_plate", subsample=predictions, size=dataset_size):
                return pyro.sample(self.data_name, predictive_distribution, obs=obs)
        else:
            dataset_size = self.dataset_size if self.dataset_size is not None else 1
            with pyro.poutine.scale(scale=dataset_size):
                return pyro.sample(self.data_name, predictive_distribution, obs=obs)

    def log_likelihood(self, predictions, data, aggregation_dim=None, reduction="none"):
        if aggregation_dim is not None:
            predictions = self.aggregate_predictions(predictions, aggregation_dim)
        log_probs = self.predictive_distribution(predictions).log_prob(data)
        return _reduce(log_probs, reduction)

    def error(self, predictions, data, aggregation_dim=None, reduction="none"):
        if aggregation_dim is not None:
            predictions = self.aggregate_predictions(predictions, aggregation_dim)
        errors = dist.util.sum_rightmost(self._calc_error(self._point_predictions(predictions), data), self.event_dim)
        return _reduce(errors, reduction)

    def sample(self, predictions, sample_shape=torch.Size()):
        return self.predictive_distribution(predictions).sample(sample_shape)

    def predictive_distribution(self, predictions):
        return self.batch_predictive_distribution(predictions).to_event(self.event_dim)

    def batch_predictive_distribution(self, predictions):
        """Returns a batched object of predictive distributions."""
        raise NotImplementedError

    def aggregate_predictions(self, predictions, dim=0):
        """Aggregates multiple samples of predictions, e.g. averages for Gaussian or probabilities."""
        raise NotImplementedError

    def _point_predictions(self, predictions):
        """Point predictions without noise, e.g. hard class labels for Bernoulli or Categorical."""
        raise NotImplementedError

    def _calc_error(self, point_predictions, data):
        """Typical error measure, e.g. squared errors for Gaussians or number of mis-classifications for Categorical."""
        raise NotImplementedError


class _Discrete(Likelihood):
    """Discrete base class that unifies logic for Bernoulli and Categorical likelihood classes."""

    def __init__(self, dataset_size, logit_predictions=True, event_dim=0, name="", data_name="data"):
        super().__init__(dataset_size, event_dim=event_dim, name=name, data_name=data_name)
        self.logit_predictions = logit_predictions

    def base_dist(self, probs=None, logits=None):
        raise NotImplementedError

    def batch_predictive_distribution(self, predictions):
        return self.base_dist(logits=predictions) if self.logit_predictions else self.base_dist(probs=predictions)

    def _calc_error(self, point_predictions, data):
        return point_predictions.ne(data).float()

    def aggregate_predictions(self, predictions, dim=0):
        probs = dist_utils.logits_to_probs(predictions, is_binary=self.is_binary) if self.logit_predictions else predictions
        avg_probs = probs.mean(dim)
        return dist_utils.probs_to_logits(avg_probs, is_binary=self.is_binary) if self.logit_predictions else avg_probs

    @property
    def is_binary(self):
        raise NotImplementedError


class Bernoulli(_Discrete):
    """Bernoulli likelihood for binary observations."""

    base_dist = dist.Bernoulli

    def _point_predictions(self, predictions):
        return predictions.gt(0.) if self.logit_predictions else predictions.gt(0.5)

    @property
    def is_binary(self):
        return True


class Categorical(_Discrete):
    """Categorical likelihood for multi-class observations."""

    base_dist = dist.Categorical

    def _point_predictions(self, predictions):
        return predictions.argmax(-1)

    @property
    def is_binary(self):
        return False


class Gaussian(Likelihood):
    """Base class for Gaussian likelihoods."""

    def __init__(self, dataset_size, event_dim=1, name="", data_name="data"):
        super().__init__(dataset_size, event_dim=event_dim, name=name, data_name=data_name)
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
    """Heteroskedastic Gaussian likelihood, i.e. Gaussian with data-dependent observation noise that is assumed to be
    part of the predictions. For d-dimensional observations, the predictions are expected to be 2d, with the tensor
    of predictions being split in the middle along the final event dim and the first half corresponding to predicted
    means and the second half to the standard deviations (which may be negative, in which case they are passed
    through a softplus function).

    :param bool positive_scale: Whether the predicted scales can be assumed to be positive."""

    def __init__(self, dataset_size, positive_scale=False, event_dim=1, name="", data_name="data"):
        super().__init__(dataset_size, event_dim=event_dim, name=name, data_name=data_name)
        self.positive_scale = positive_scale

    def aggregate_predictions(self, predictions, dim=0):
        """Aggregates multiple predictions for the same data by averaging them according to their predicted noise.
        Means with lower predicted noise are given higher weight in the average. Predictive variance is the variance
        of the means plus the average predicted variance."""
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
    """Homeskedastic Gaussian likelihood, i.e. a likelihood that assumes the noise to be data-independent. The scale
    or precision may be a distribution, i.e. be unknown and have a prior placed on it for it to be inferred or be a
    PyroParameter in order to be learnable.

    :param scale: tensor, parameter or prior distribution for the scale. Mutually exclusive with precision.
    :param precision: tensor, parameter or prior distribution for the precision. Mutually exclusive with scale."""

    def __init__(self, dataset_size, scale=None, precision=None, event_dim=1, name="", data_name="data"):
        super().__init__(dataset_size, event_dim=event_dim, name=name, data_name=data_name)
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
        """Aggregates multiple predictions for the same data by averaging them. Predictive variance is the variance
         of the predictions plus the known variance term."""
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
