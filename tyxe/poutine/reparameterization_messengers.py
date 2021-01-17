from functools import update_wrapper
from weakref import WeakValueDictionary, WeakKeyDictionary

import torch
import torch.nn.functional as F

import pyro.distributions as dist
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import effectful


__all__ = [
    "LocalReparameterizationMessenger",
    "FlipoutMessenger"
]


def _get_base_dist(distribution):
    while isinstance(distribution, dist.Independent):
        distribution = distribution.base_dist
    return distribution


def _is_reparameterizable(distribution):
    if distribution is None:
        # bias terms may be None, which does not prevent reparameterization
        return True
    return isinstance(_get_base_dist(distribution), (dist.Normal, dist.Delta))


def _get_loc_var(distribution):
    if distribution is None:
        return None, None
    if torch.is_tensor(distribution):
        # distribution might be a pyro param, which is equivalent to a delta distribution
        return distribution, torch.zeros_like(distribution)
    distribution = _get_base_dist(distribution)
    return distribution.mean, distribution.variance


class _ReparameterizationMessenger(Messenger):
    """Base class for reparameterization of sampling sites where a transformation of a stochastic by a deterministic
    variable allows for analytically calculating (or approximation) the distribution of the result and sampling
    the result instead of the original stochastic variable. See subclasses for examples.

    Within the context of this messenger, functions in the REPARAMETERIZABLE_FUNCTIONS attribute will have the
    outputs sampled instead of the inputs to the weight and bias attributes. This can reduce gradient noise. For now,
    reparameterization is limited to F.linear and F.conv, which are used by the corresponding nn.Linear and nn.Conv
    modules in pytorch."""

    # TODO check if transposed convolutions could be added as well, might be useful for Bayesian conv VAEs
    REPARAMETERIZABLE_FUNCTIONS = ["linear", "conv1d", "conv2d", "conv3d"]

    def __init__(self, reparameterizable_functions=None):
        super().__init__()
        if reparameterizable_functions is None:
            reparameterizable_functions = self.REPARAMETERIZABLE_FUNCTIONS
        elif isinstance(reparameterizable_functions, str):
            reparameterizable_functions = [reparameterizable_functions]
        elif isinstance(reparameterizable_functions, (list, tuple)):
            reparameterizable_functions = list(reparameterizable_functions)
        else:
            raise ValueError(f"Unrecognized type for argument 'reparameterizable_functions. Must be str, list or "
                             f"None, but go '{reparameterizable_functions.__class__.__name__}'.")
        self.reparameterizable_functions = reparameterizable_functions

    def __enter__(self):
        # deps maps sampled tensors to distributon object to check if local reparameterization is possible.
        # I'm using a weakref dictionary here for memory efficiency -- a standard dict would create references to all
        # kinds of intermediate tensors, preventing them from being garbage collected. This would be a problem if the
        # Messenger is used as a context outside of a training loop. Ideally I would like to use a WeakKeyDictionary,
        # since I would expect that the samples from the distribution are much less likely to be kept around than the
        # distribution object itself. I'm using id(tensor) as dictionary keys in order to avoid creating references to
        # the samples from the distributions. However this still means that the self.deps dictionary will keep growing
        # if the distribution objects from the model/guide are kept around.
        self.deps = WeakValueDictionary()
        self.original_fns = [getattr(F, name) for name in self.reparameterizable_functions]
        self._make_reparameterizable_functions_effectful()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._reset_reparameterizable_functions()
        del self.deps
        del self.original_fns
        return super().__exit__(exc_type, exc_val, exc_tb)

    def _make_reparameterizable_functions_effectful(self):
        for name, fn in zip(self.reparameterizable_functions, self.original_fns):
            effectful_fn = update_wrapper(effectful(fn, type="reparameterizable"), fn)
            setattr(F, name, effectful_fn)

    def _reset_reparameterizable_functions(self):
        for name, fn in zip(self.reparameterizable_functions, self.original_fns):
            setattr(F, name, fn)

    def _pyro_post_sample(self, msg):
        if id(msg["value"]) not in self.deps:
            self.deps[id(msg["value"])] = msg["fn"]

    def _pyro_reparameterizable(self, msg):
        if msg["fn"].__name__ not in self.reparameterizable_functions:
            return

        if msg["done"]:
            raise ValueError(f"Trying to reparameterize a {msg['fn'].__name__} site that has already been processed. "
                             f"Did you use multiple reparameterization messengers for the same function?")

        args = list(msg["args"])
        kwargs = msg["kwargs"]
        x = kwargs.pop("input", None) or args.pop(0)
        # if w is in args, so must have been x, therefore w will now be the first argument in args if not in kwargs
        w = kwargs.pop("weight", None) or args.pop(0)
        # bias might be None, so check explicitly if it's in kwargs -- if it is positional, x and w
        # must have been positional arguments as well
        b = kwargs.pop("bias") if "bias" in kwargs else args.pop(0)
        if id(w) in self.deps:
            w_fn = self.deps[id(w)]
            b_fn = self.deps[id(b)] if b is not None else None
            if torch.is_tensor(x) and _is_reparameterizable(w_fn) and _is_reparameterizable(b_fn):
                msg["value"] = self._reparameterize(msg, x, w_fn, w, b_fn, b, *args, **kwargs)
                msg["done"] = True

    def _reparameterize(self, msg, x, w_loc, w_var, b_loc, b_var, *args, **kwargs):
        raise NotImplementedError


class LocalReparameterizationMessenger(_ReparameterizationMessenger):
    """Implements local reparameterization: https://arxiv.org/abs/1506.02557"""

    def _reparameterize(self, msg, x, w_fn, w, b_fn, b, *args, **kwargs):
        w_loc, w_var = _get_loc_var(w_fn)
        b_loc, b_var = _get_loc_var(b_fn)
        loc = msg["fn"](x, w_loc, b_loc, *args, **kwargs)
        var = msg["fn"](x.pow(2), w_var, b_var, *args, **kwargs)
        # ensure positive variances to avoid NaNs when taking square root
        var = var + var.lt(0).float().mul(var.abs() + 1e-6).detach()
        scale = var.sqrt()
        return dist.Normal(loc, scale).rsample()


def _pad_right_like(tensor1, tensor2):
    while tensor1.ndim < tensor2.ndim:
        tensor1 = tensor1.unsqueeze(-1)
    return tensor1


def _rand_signs(*args, **kwargs):
    return torch.rand(*args, **kwargs).gt(0.5).float().mul(2).sub(1)


class FlipoutMessenger(_ReparameterizationMessenger):
    """Implements flipout: https://arxiv.org/abs/1803.04386"""

    FUNCTION_RANKS = {"linear": 1, "conv1d": 2, "conv2d": 3, "conv3d": 4}

    def _reparameterize(self, msg, x, w_fn, w, b_fn, b, *args, **kwargs):
        fn = msg["fn"]
        w_loc, _ = _get_loc_var(w_fn)
        loc = fn(x, w_loc, None, *args, **kwargs)

        # x might be one dimensional for a 1-d input with a single datapoint to F.linear, F.conv always has a batch dim
        batch_shape = x.shape[:-self.FUNCTION_RANKS[fn.__name__]] if x.ndim > 1 else tuple()
        # w might be 1-d for F.linear for a 0-d output
        output_shape = (w_loc.shape[0],) if w_loc.ndim > 1 else tuple()
        input_shape = (w_loc.shape[1],) if w_loc.ndim > 1 else (w_loc.shape[0],)

        if not hasattr(w, "sign_input"):
            w.sign_input = _pad_right_like(_rand_signs(batch_shape + input_shape, device=loc.device), x)
            w.sign_output = _pad_right_like(_rand_signs(batch_shape + output_shape, device=loc.device), x)

        w_perturbation = w - w_loc
        perturbation = fn(x * w.sign_input, w_perturbation, None, *args, **kwargs) * w.sign_output

        output = loc + perturbation
        if b is not None:
            b_loc, b_var = _get_loc_var(b_fn)
            bias = _pad_right_like(dist.Normal(b_loc, b_var.sqrt()).rsample(batch_shape), output)
            output += bias
        return output
