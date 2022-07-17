from abc import ABCMeta, abstractmethod

import torch.nn.init as nn_init

import pyro.distributions as dist                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
from pyro.nn.module import PyroSample, PyroParam


from . import util


def _make_expose_fn(hide_modules, expose_modules, hide_module_types, expose_module_types,
                    hide_parameters, expose_parameters, hide, expose):
    if expose_modules is None:
        expose_modules = []
    else:
        expose_all = False

    if hide_modules is None:
        hide_modules = []
    else:
        expose_all = True

    if expose_module_types is None:
        expose_module_types = tuple()
    else:
        expose_all = False

    if hide_module_types is None:
        hide_module_types = tuple()
    else:
        expose_all = True

    if expose_parameters is None:
        expose_parameters = []
    else:
        expose_all = False

    if hide_parameters is None:
        hide_parameters = []
    else:
        expose_all = True

    if expose is None:
        expose = []
    else:
        expose_all = False

    if hide is None:
        hide = []
    else:
        expose_all = True

    if not set(hide_modules).isdisjoint(set(expose_modules)):
        raise ValueError("Cannot hide and expose a module.")

    if not set(hide_parameters).isdisjoint(set(expose_parameters)):
        raise ValueError("Cannot hide and expose a parameter type.")

    if not set(hide).isdisjoint(set(expose)):
        raise ValueError("Cannot hide and expose a parameter.")

    def expose_fn(module, param_name):
        if param_name in hide:
            return False
        if param_name in expose:
            return True

        param_suffix = param_name.rsplit(".")[-1]
        if param_suffix in hide_parameters:
            return False
        if param_suffix in expose_parameters:
            return True

        if isinstance(module, hide_module_types):
            return False
        if isinstance(module, expose_module_types):
            return True

        if module in hide_modules:
            return False
        if module in expose_modules:
            return True

        return expose_all

    return expose_fn


class Prior(metaclass=ABCMeta):
    """Base class for TyXe's BNN priors that helps with replacing nn.Parameter attributes on PyroModule objects
    with PyroSamples via its apply_ method or updating them via update_ and handles logic for excluding some parameters
    from having priors based on them via the hide/exclude arguments of the init method. Subclasses must implement
    a prior_dist method that returns a distribution object given a parameter name, module and nn.Parameter object."""

    def __init__(self, hide_all=False, expose_all=True,
                 hide_modules=None, expose_modules=None,
                 hide_module_types=None, expose_module_types=None,
                 hide_parameters=None, expose_parameters=None,
                 hide=None, expose=None,
                 hide_fn=None, expose_fn=None):
        """Hides/exposes parameter attributes from/to being replaced by PyroSamples. The options are:
        * all: hides/exposes all parameters. expose_all must be set to False for using any of the other options.
        * modules: nn.Modules object that are part of the net being passed in apply_.
        * module_types: tuple of classes inheriting from nn.Module, e.g. nn.Linear.
        * parameters: list of parameter attribute names, e.g. "weight" for hiding/exposing the weight attribute of
            an nn.Linear module.
        * hide/expose: list of full parameter names, e.g. "0.weight" for a nn.Sequential net where the first layer is a
            a nn.Conv or nn.Linear module that has a weight attribute.
        * fn: function that returns True or False given a module and param_name string."""
        if hide_all:
            self.expose_fn = lambda module, name: False
        elif expose_fn is not None:
            self.expose_fn = expose_fn
        elif hide_fn is not None:
            self.expose_fn = lambda module, name: not hide_fn(module, name)
        elif expose_all:
            self.expose_fn = lambda module, name: True
        else:
            self.expose_fn = _make_expose_fn(
                hide_modules, expose_modules, hide_module_types, expose_module_types,
                hide_parameters, expose_parameters, hide, expose)

    def apply_(self, net):
        """"Replaces all nn.Parameter attributes on a given PyroModule net according to the hide/expose logic and
        the classes' prior_dist method."""
        for module_name, module in net.named_modules():
            for param_name, param in list(module.named_parameters(recurse=False)):
                full_name = module_name + "." + param_name
                if self.expose_fn(module, full_name):
                    prior_dist = self.prior_dist(full_name, module, param).expand(param.shape).to_event(param.dim())
                    setattr(module, param_name, PyroSample(prior_dist))
                else:
                    setattr(module, param_name, PyroParam(param.data.detach()))

    def update_(self, net):
        """Replaces PyroSample attributes on a given PyroModule net according to the hide/expose logic and
        the classes' prior_dist method."""
        for module_name, module in net.named_modules():
            for site_name, site in list(util.named_pyro_samples(module, recurse=False)):
                full_name = module_name + "." + site_name
                # See change in DictPrior as an alternative
                # if type(self)== DictPrior: 
                #     full_name = 'net.' + full_name
                if self.expose_fn(module, full_name):
                    prior_dist = self.prior_dist(full_name, module, site)
                    setattr(module, site_name, PyroSample(prior_dist))

    @abstractmethod
    def prior_dist(self, name, module, param):
        pass


class IIDPrior(Prior):
    """Independent identically distributed prior that is the same across all sites. Intended to be used with
    one-dimensional distribution that can be extended to the shape of each site, e.g. dist.Normal."""

    def __init__(self, distribution, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distribution = distribution

    def prior_dist(self, name, module, param):
        return self._distribution


class LayerwiseNormalPrior(Prior):
    """Normal prior with module-dependent variance to preserve the variance of an input passed through the layer.
    "radford" sets the variance to the inverse of the number of inputs, "kaiming" multiplies this with an additional
    gain factor depending on the nonlinearity and "xavier" to the inverse of the average of the number of inputs
    and outputs (the latter correspond to weight initialization methods for deterministic neural networks)."""

    def __init__(self, method="radford", nonlinearity="relu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if method not in ("radford", "xavier", "kaiming"):
            raise ValueError(f"variance must be one of ('radford', 'xavier', 'kaiming'), but is {method}")
        self.method = method
        self.nonlinearity = nonlinearity

    def prior_dist(self, name, module, param):
        module_nonl = self.nonlinearity if isinstance(self.nonlinearity, str) else self.nonlinearity.get(module)
        gain = nn_init.calculate_gain(module_nonl) if module_nonl is not None else 1.
        std = util.calculate_prior_std(self.method, param, gain)
        return dist.Normal(0., std)


class DictPrior(Prior):
    """Dictionary of prior distributions mapping parameter names as in module.named_parameters() to distribution
    objects."""

    def __init__(self, prior_dict,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_dict = prior_dict

    def prior_dist(self, name, module, param):
        return self.prior_dict[name]

class LambdaPrior(Prior):
    """Utility class to avoid implementing a prior class for a given function."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn

    def prior_dist(self, name, module, param):
        return self.fn(name, module, param)
