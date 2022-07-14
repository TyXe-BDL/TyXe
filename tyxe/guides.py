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


def _get_base_dist(distribution):
    while isinstance(distribution, dist.Independent):
        distribution = distribution.base_dist
    return distribution


def init_to_constant(site, c):
    """Helper function to set site value to a constant value."""
    site_fn = site["fn"]
    value = torch.full_like(site_fn.sample(), c)
    if hasattr(site_fn, "_validate_sample"):
        site_fn._validate_sample(value)
    return value


def init_to_zero(site):
    """Helper function to set site value to 0."""
    return init_to_constant(site, 0.)


def init_to_sample(site, distribution):
    """Helper function to set site value to a sample from some given distribution."""
    value = distribution.expand(site["fn"].event_shape).sample().detach()
    t = transform_to(site["fn"].support)
    return t(value)


def init_to_normal(site, loc=0., std="xavier", gain=1.):
    """Helper function to set site value to a sample from a normal distribution with variance according to
    xavier/kaiming/radford neural network weight initialization methods."""
    if isinstance(std, str):
        std = util.calculate_prior_std(std, site["fn"].sample(), gain=gain)
    return init_to_sample(site, dist.Normal(loc, std))


def init_to_normal_xavier(site):
    return init_to_normal(site, std="xavier")


def init_to_normal_radford(site):
    return init_to_normal(site, std="radford")


def init_to_normal_kaiming(site, gain=1.):
    return init_to_normal(site, std="kaiming", gain=gain)


class PretrainedInitializer:
    """Utility class for setting the values of a site to known constants, e.g. from a trained neural network.

    :param dict values: dictionary of parameter values, mapping names of sites to tensors"""

    def __init__(self, values):
        self.values = values

    def __call__(self, site):
        return self.values[site["name"]]

    @classmethod
    def from_net(cls, net, prefix="net"):
        """Alternative init method for instantiating the class from the parameter values of an nn.Module.
        
        :param module: nn.Module to extract parameters from
        :param string prefix: Prefix value to pass to the modules `named_parameters` function
        
        :rtype: PretrainedInitializer
        """
        values = {}
        for name, parameter in net.named_parameters(prefix):
            values[name] = parameter.data.clone()
        return cls(values)


class AutoNormal(ag.AutoGuide):
    """Variant of pyro.infer.autoguide.AutoNormal. Samples sites from TransformedDistribution objects of normal
    distributions to allow for calculating KL divergences in closed form. Further makes training means or variances
    optional as well as allowing for variances to be capped at some upper limit. Provides a helper function for
    returning a subset of all site distributions with the parameters detached to be used as priors in variational
    continual learning.

    :param module: PyroModule or pyro model to perform inference in.
    :param callable init_loc_fn: function that sets the means of variational distributions of each site.
    :param float init_scale: initial standard deviation of the variational distributions.
    :param bool train_loc: Whether the variational means should be learnable.
    :param bool train_scale: Whether the variational standard deviations should be learnable.
    :param float max_guide_scale: Optional upper limit on the variational standard deviations."""

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
            constrained_value = site["value"]
            unconstrained_value = biject_to(site["fn"].support).inv(constrained_value)
            if self.train_loc:
                unconstrained_value = pynn.PyroParam(unconstrained_value)
            ag.guides.deep_setattr(self, name + ".loc", unconstrained_value)
            if isinstance(self.init_scale, numbers.Real):
                scale_value = torch.full_like(site["value"], self.init_scale)
            elif isinstance(self.init_scale, str):
                scale_value = torch.full_like(site["value"], util.calculate_prior_std(self.init_scale, site["value"]))
            else:
                scale_value = self.init_scale[site["name"]]
            scale_constraint = constraints.positive if self.max_guide_scale is None else constraints.interval(0., self.max_guide_scale)
            scale = pynn.PyroParam(scale_value, constraint=scale_constraint) if self.train_scale else scale_value   
            ag.guides.deep_setattr(self, name + ".scale", scale)
            
    def get_loc(self, site_name):
        return pyro.util.deep_getattr(self, site_name + ".loc")

    def get_scale(self, site_name):
        return pyro.util.deep_getattr(self, site_name + ".scale")

    def get_detached_distributions(self, site_names=None):
        """Returns a dictionary mapping the site names to their variational posteriors. All parameters are detached."""
        if site_names is None:
            site_names = list(name for name, _ in self.prototype_trace.iter_stochastic_nodes())

        result = dict()
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            if name not in site_names:
                continue
            loc = self.get_loc(name).detach().clone()
            scale = self.get_scale(name).detach().clone()
            fn = dist.Normal(loc, scale).to_event(max(loc.dim(), scale.dim()))
            base_fn = _get_base_dist(site["fn"])
            if base_fn.support is not dist.constraints.real:
                fn = dist.TransformedDistribution(fn, biject_to(base_fn.support))
            result[name] = fn
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
                fn = dist.Normal(loc, scale).to_event(site["fn"].event_dim)
                base_fn = _get_base_dist(site["fn"])
                if base_fn.support is not dist.constraints.real:
                    fn = dist.TransformedDistribution(fn, biject_to(base_fn.support))
                result[name] = pyro.sample(name, fn)
        return result

    def median(self, *args, **kwargs):
        return {site["name"]: biject_to(site["fn"].support)(self.get_loc(site["name"]))
                for site in self.prototype_trace.iter_stochastic_nodes()}
