import pytest

from functools import partial

import torch

import pyro
import pyro.distributions as dist
from pyro.infer import TraceMeanField_ELBO
from pyro.nn import PyroModule, PyroSample

import tyxe


@pytest.fixture(autouse=True)
def setup():
    pyro.clear_param_store()


def test_constant_kl():
    model = lambda: pyro.sample("a", dist.Normal(0, 1.))
    guide = tyxe.guides.AutoNormal(model)
    elbo = TraceMeanField_ELBO()
    assert elbo.loss(model, guide) == elbo.loss(model, guide)


def test_auto_normal():
    model = lambda: pyro.sample("a", dist.Normal(0., 1.))
    guide = tyxe.guides.AutoNormal(model)
    tr = pyro.poutine.trace(guide).get_trace()
    fn = tr.nodes["a"]["fn"]
    assert isinstance(fn, dist.Normal)
    assert fn.scale.isclose(torch.tensor(0.1))
    assert fn.loc.requires_grad
    assert fn.scale.requires_grad


def test_auto_normal_constrained():
    model = lambda: pyro.sample("a", dist.Gamma(1., 1.))
    guide = tyxe.guides.AutoNormal(model)
    tr = pyro.poutine.trace(guide).get_trace()
    fn = tr.nodes["a"]["fn"]
    assert isinstance(fn, dist.TransformedDistribution)
    assert isinstance(fn.base_dist, dist.Normal)
    assert fn.base_dist.scale.isclose(torch.tensor(0.1))
    assert fn.base_dist.loc.requires_grad
    assert fn.base_dist.scale.requires_grad


def test_auto_normal_constant_loc():
    model = lambda: pyro.sample("a", dist.Normal(0., 1.))
    guide = tyxe.guides.AutoNormal(model, train_loc=False)
    guide()
    assert not guide.a.loc.requires_grad


def test_auto_normal_constant_scale():
    model = lambda: pyro.sample("a", dist.Normal(0., 1.))
    guide = tyxe.guides.AutoNormal(model, train_scale=False)
    guide()
    assert not guide.a.scale.requires_grad


def test_auto_normal_init_scale():
    model = lambda: pyro.sample("a", dist.Normal(0., 1.))
    guide = tyxe.guides.AutoNormal(model, init_scale=1e-2)
    guide()
    assert guide.a.scale.isclose(torch.tensor(1e-2))


def test_auto_normal_max_scale():
    model = lambda: pyro.sample("a", dist.Normal(0., 1.))
    guide = tyxe.guides.AutoNormal(model, init_scale=1e-2, max_guide_scale=1e-3)
    guide()
    assert guide.a.scale.isclose(torch.tensor(1e-3))


def test_auto_normal_detached_distributions():
    model = lambda: pyro.sample("a", dist.Normal(0., 1.))
    guide = tyxe.guides.AutoNormal(model)
    guide()
    fn = guide.get_detached_distributions()["a"]
    assert isinstance(fn, dist.Normal)
    assert fn.scale.isclose(torch.tensor(0.1))
    assert not fn.loc.requires_grad
    assert not fn.scale.requires_grad


def test_constant_init():
    model = lambda: pyro.sample("a", dist.Normal(torch.ones(3), 1.))
    guide = tyxe.guides.AutoNormal(model, init_loc_fn=partial(tyxe.guides.init_to_constant, c=2))
    guide()
    assert guide.a.loc.eq(2).all()


def test_zero_init():
    model = lambda: pyro.sample("a", dist.Normal(torch.ones(3, 2), 1.))
    guide = tyxe.guides.AutoNormal(model, init_loc_fn=tyxe.guides.init_to_zero)
    guide()
    assert guide.a.loc.eq(0).all()


def test_sample_init():
    model = lambda: pyro.sample("a", dist.Normal(torch.ones(10000), 1.).to_event())
    guide = tyxe.guides.AutoNormal(
        model, init_loc_fn=partial(tyxe.guides.init_to_sample, distribution=dist.Normal(0, 1)))
    guide()
    assert guide.a.loc.mean().isclose(torch.tensor(0.), atol=3e-2).item()
    assert guide.a.loc.std().isclose(torch.tensor(1.), atol=3e-2).item()


def test_xavier_init():
    model = lambda: pyro.sample("a", dist.Normal(torch.ones(50, 150), 1.).to_event())
    guide = tyxe.guides.AutoNormal(model, init_loc_fn=tyxe.guides.init_to_normal_kaiming)
    guide()
    assert guide.a.loc.std().isclose(torch.tensor(0.1), atol=3e-2).item()


def test_radford_init():
    model = lambda: pyro.sample("a", dist.Normal(torch.ones(50, 144), 1.).to_event())
    guide = tyxe.guides.AutoNormal(model, init_loc_fn=tyxe.guides.init_to_normal_kaiming)
    guide()
    assert guide.a.loc.std().isclose(torch.tensor(144 ** -0.5), atol=3e-2).item()


def test_kaiming_init():
    model = lambda: pyro.sample("a", dist.Normal(torch.ones(100, 100), 1.).to_event())
    guide = tyxe.guides.AutoNormal(model, init_loc_fn=partial(tyxe.guides.init_to_normal_kaiming, gain=2.))
    guide()
    assert guide.a.loc.std().isclose(torch.tensor(0.2), atol=3e-2).item()


def test_pretrained_init():
    model = lambda: pyro.sample("a", dist.Normal(torch.ones(5), 1.).to_event())
    mean_init = torch.randn(5)
    guide = tyxe.guides.AutoNormal(model, init_loc_fn=tyxe.guides.PretrainedInitializer({"a": mean_init}))
    guide()
    assert guide.a.loc.eq(mean_init).all().item()


def test_pretrained_from_net_init():
    l = torch.nn.Linear(3, 2, bias=False)
    model = PyroModule[torch.nn.Linear](3, 2, bias=False)
    model.weight = PyroSample(dist.Normal(torch.zeros_like(model.weight), torch.ones_like(model.weight)))
    guide = tyxe.guides.AutoNormal(model, init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(l, prefix=""))
    guide(torch.randn(3))
    assert guide.weight.loc.eq(l.weight).all().item()
