from collections import OrderedDict
import copy
from functools import reduce
from operator import mul, itemgetter
from warnings import warn

import torch

import pyro.util
import pyro.infer.autoguide.guides
import pyro.nn.module as pyromodule

def deep_hasattr(obj, name):
    warn('deep_hasattr is deprecated.', DeprecationWarning, stacklevel=2)
    try:
        pyro.util.deep_getattr(obj, name)
        return True
    except AttributeError:
        return False


def deep_setattr(obj, key, val):
    warn('deep_setattr is deprecated.', DeprecationWarning, stacklevel=2)
    return pyro.infer.autoguide.guides.deep_setattr(obj, key, val)

def deep_getattr(obj, name):
    warn('deep_getattr is deprecated.', DeprecationWarning, stacklevel=2)
    return pyro.util.deep_getattr(obj, name)


def to_pyro_module_(m, name="", recurse=True):
    """
    Same as `pyro.nn.modules.to_pyro_module_` except that it also accepts a name argument and returns the modified
    module following the convention in pytorch for inplace functions.
    """
    if not isinstance(m, torch.nn.Module):
        raise TypeError("Expected an nn.Module instance but got a {}".format(type(m)))

    if isinstance(m, pyromodule.PyroModule):
        if recurse:
            for name, value in list(m._modules.items()):
                to_pyro_module_(value)
                setattr(m, name, value)
        return

    # Change m's type in-place.
    m.__class__ = pyromodule.PyroModule[m.__class__]
    m._pyro_name = name
    m._pyro_context = pyromodule._Context()
    m._pyro_params = OrderedDict()
    m._pyro_samples = OrderedDict()

    # Reregister parameters and submodules.
    for name, value in list(m._parameters.items()):
        setattr(m, name, value)
    for name, value in list(m._modules.items()):
        if recurse:
            to_pyro_module_(value)
        setattr(m, name, value)

    return m


def to_pyro_module(m, name="", recurse=True):
    return to_pyro_module_(copy.deepcopy(m), name, recurse)


def named_pyro_samples(pyro_module, prefix='', recurse=True):
    yield from pyro_module._named_members(lambda module: module._pyro_samples.items(), prefix=prefix, recurse=recurse)


def pyro_sample_sites(pyro_module, prefix='', recurse=True):
    yield from map(itemgetter(0), named_pyro_samples(pyro_module, prefix=prefix, recurse=recurse))


def prod(iterable, initial_value=1):
    return reduce(mul, iterable, initial_value)


def fan_in_fan_out(weight):
    # this holds for linear and conv layers, but check e.g. transposed conv
    fan_in = prod(weight.shape[1:])
    fan_out = weight.shape[0]
    return fan_in, fan_out


def calculate_prior_std(method, weight, gain=1., mode="fan_in"):
    fan_in, fan_out = fan_in_fan_out(weight)
    if method == "radford":
        std = fan_in ** -0.5
    elif method == "xavier":
        std = gain * (2 / (fan_in + fan_out)) ** 0.5
    elif method == "kaiming":
        fan = fan_in if mode == "fan_in" else fan_out
        std = gain * fan ** -0.5
    else:
        raise ValueError(f"Invalid method: '{method}'. Must be one of ('radford', 'xavier', 'kaiming'.")
    return torch.tensor(std, device=weight.device)
