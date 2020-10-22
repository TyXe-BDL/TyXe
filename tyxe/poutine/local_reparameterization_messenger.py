from functools import update_wrapper
from weakref import WeakValueDictionary

import torch
import torch.nn.functional as F

import pyro.distributions as dist
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import effectful


__all__ = ["LocalReparameterizationMessenger"]


# TODO check if transposed convolutions could be added as well, might be useful for Bayesian conv VAEs
LINEAR_FUNS = ["linear", "conv1d", "conv2d", "conv3d"]


def _get_base_dist(distribution):
    while isinstance(distribution, dist.Independent):
        distribution = distribution.base_dist
    return distribution


def _is_gaussian(distribution):
    return isinstance(_get_base_dist(distribution), dist.Normal)


def _get_loc_var(distribution):
    if distribution is None:
        return None, None
    distribution =  _get_base_dist(distribution)
    return distribution.loc, distribution.scale.pow(2)


class LocalReparameterizationMessenger(Messenger):

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
        self.original_fns = [getattr(F, fn) for fn in LINEAR_FUNS]
        self.make_effectful()
        return super(LocalReparameterizationMessenger, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset_fns()
        del self.deps
        del self.original_fns
        return super(LocalReparameterizationMessenger, self).__exit__(exc_type, exc_val, exc_tb)

    def make_effectful(self):
        for name, fn in zip(LINEAR_FUNS, self.original_fns):
            effectful_fn = update_wrapper(effectful(fn, type="linear"), fn)
            setattr(F, name, effectful_fn)

    def reset_fns(self):
        for name, fn in zip(LINEAR_FUNS, self.original_fns):
            setattr(F, name, fn)

    def _pyro_post_sample(self, msg):
        if id(msg["value"]) not in self.deps:
            self.deps[id(msg["value"])] = msg["fn"]

    def _pyro_linear(self, msg):
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
            if torch.is_tensor(x) and _is_gaussian(w_fn) and (b is None or _is_gaussian(b_fn)):
                w_loc, w_var = _get_loc_var(w_fn)
                b_loc, b_var = _get_loc_var(b_fn)
                loc = msg["fn"](x, w_loc, b_loc, *args, **kwargs)
                var = msg["fn"](x.pow(2), w_var, b_var, *args, **kwargs)
                # ensure positive variances to avoid NaNs when taking square root
                var = var + var.lt(0).float().mul(var.abs() + 1e-6).detach()
                scale = var.sqrt()
                msg["value"] = dist.Normal(loc, scale).rsample()
                msg["done"] = True
