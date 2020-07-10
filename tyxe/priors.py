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


# TODO make sure that distributions live on the correct device? Or is that the user's responsibility? It would probably
# be easiest to just handle this in the BNN convenience function and not touch anything here to avoid unexpected
# behaviour
class Prior(metaclass=ABCMeta):

    def __init__(self, hide_all=False, expose_all=True,
                 hide_modules=None, expose_modules=None,
                 hide_module_types=None, expose_module_types=None,
                 hide_parameters=None, expose_parameters=None,
                 hide=None, expose=None,
                 hide_fn=None, expose_fn=None):
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
        for module_name, module in net.named_modules():
            for param_name, param in list(module.named_parameters(recurse=False)):
                full_name = module_name + "." + param_name
                if self.expose_fn(module, full_name):
                    prior_dist = self.prior_dist(full_name, module, param).expand(param.shape).to_event(param.dim())
                    setattr(module, param_name, PyroSample(prior_dist))
                else:
                    setattr(module, param_name, PyroParam(param.data.detach()))

    def update_(self, net):
        for module_name, module in net.named_modules():
            for site_name, site in list(util.named_pyro_samples(module, recurse=False)):
                full_name = module_name + "." + site_name
                if self.expose_fn(module, full_name):
                    prior_dist = self.prior_dist(full_name, module, site)
                    setattr(module, site_name, PyroSample(prior_dist))

    @abstractmethod
    def prior_dist(self, name, module, param):
        pass


class IIDPrior(Prior):

    def __init__(self, distribution, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distribution = distribution

    def prior_dist(self, name, module, param):
        return self._distribution


# TODO add way of passing kwargs, i.e. mode for kaiming init, param for leaky_relu nonlinearities
class LayerwiseNormalPrior(Prior):

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

    def __init__(self, prior_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_dict = prior_dict

    def prior_dist(self, name, module, param):
        return self.prior_dict[name]


class LambdaPrior(Prior):

    def __init__(self, fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn

    def prior_dist(self, name, module, param):
        return self.fn(name, module, param)
