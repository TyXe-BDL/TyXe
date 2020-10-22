from contextlib import ExitStack
import numbers

import torch
from torch.distributions import biject_to, transform_to

import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
import pyro.nn as pynn
import pyro.infer.autoguide as ag
import pyro.infer.autoguide.initialization as ag_init
import pyro.util as pyutil


from . import util


def init_to_constant(site, c):
    site_fn = site["fn"]
    value = torch.full_like(site_fn.sample(), c)
    if hasattr(site_fn, "_validate_sample"):
        site_fn._validate_sample(value)
    return value


def init_to_zero(site):
    return init_to_constant(site, 0.)


def init_to_sample(site, distribution):
    value = distribution.expand(site["fn"].event_shape).sample().detach()
    t = transform_to(site["fn"].support)
    return t(value)


def init_to_normal(site, loc=0., std="xavier"):
    if isinstance(std, str):
        std = util.calculate_prior_std(std, site["fn"].sample())
    return init_to_sample(site, dist.Normal(loc, std))


def init_to_normal_xavier(site):
    return init_to_normal(site, std="xavier")


def init_to_normal_radford(site):
    return init_to_normal(site, std="radford")


def init_to_normal_kaiming(site):
    return init_to_normal(site, std="kaiming")


class SitewiseInitializer:

    def __init__(self, values):
        self.values = values

    def __call__(self, site):
        # return self.values.get(site["name"], ag_init.init_to_median(site))
        return self.values[site["name"]]

    @classmethod
    def from_net(cls, net, prefix=""):
        values = {}
        for name, parameter in net.named_parameters(prefix):
            values[name] = parameter.data.clone()
        return cls(values)


class ParameterwiseDiagonalNormal(ag.AutoGuide):

    # add option for making init_sd something like "radford", etc.
    def __init__(self, module, init_loc_fn=ag_init.init_to_median, init_scale=1e-1, train_loc=True, train_scale=True,
                 max_guide_scale=None):
        module = ag_init.InitMessenger(init_loc_fn)(module)
        self.init_scale = init_scale
        self.train_loc = train_loc
        self.train_scale = train_scale
        self.max_guide_scale = max_guide_scale
        super().__init__(module)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            constrained_value = pynn.PyroParam(site["value"]) if self.train_loc else site["value"]
            unconstrained_value = biject_to(site["fn"].support).inv(constrained_value)
            ag.guides._deep_setattr(self, name + ".loc", unconstrained_value)
            if isinstance(self.init_scale, numbers.Real):
                scale_value = torch.full_like(site["value"], self.init_scale)
            elif isinstance(self.init_scale, str):
                scale_value = torch.full_like(site["value"], util.calculate_prior_std(self.init_scale, site["value"]))
            else:
                scale_value = self.init_scale[site["name"]]
            scale_constraint = constraints.positive if self.max_guide_scale is None else constraints.interval(0., self.max_guide_scale)
            scale = pynn.PyroParam(scale_value, constraint=scale_constraint) if self.train_scale else scale_value
            ag.guides._deep_setattr(self, name + ".scale", scale)

    def get_loc(self, site_name):
        return pyutil.deep_getattr(self, site_name + ".loc")

    def get_scale(self, site_name):
        return pyutil.deep_getattr(self, site_name + ".scale")

    def get_detached_distributions(self, site_names):
        result = dict()
        for site in site_names:
            loc = self.get_loc(site).detach().clone()
            scale = self.get_scale(site).detach().clone()
            result[site] = dist.Normal(loc, scale).to_event(max(loc.dim(), scale.dim()))
        return result

    def forward(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates()
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])
                loc = self.get_loc(name)
                scale = self.get_scale(name)
                result[name] = pyro.sample(name, dist.Normal(loc, scale).to_event(site["fn"].event_dim))
        return result

    def median(self, *args, **kwargs):
        return {site["name"]: biject_to(site["fn"].support)(self.get_loc(site["name"]))
                for site in self.prototype_trace.iter_stochastic_nodes()}
