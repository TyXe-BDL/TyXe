import torch
import torch.nn as nn

import pyro.distributions as dist
from pyro.nn import PyroModule


import tyxe


def test_iid():
    l = PyroModule[nn.Linear](3, 2, bias=False)
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1))
    prior.apply_(l)
    p = l._pyro_samples["weight"]
    assert isinstance(p, dist.Independent)
    assert isinstance(p.base_dist, dist.Normal)
    assert p.base_dist.loc.allclose(torch.tensor(0.))
    assert p.base_dist.scale.allclose(torch.tensor(1.))


def test_layerwise_normal_kaiming():
    l = PyroModule[nn.Linear](3, 2, bias=False)
    prior = tyxe.priors.LayerwiseNormalPrior(method="kaiming")
    prior.apply_(l)
    p = l._pyro_samples["weight"]
    assert p.base_dist.scale.allclose(torch.tensor((2 / 3.) ** 0.5))


def test_layerwise_normal_radford():
    l = PyroModule[nn.Linear](3, 2, bias=False)
    prior = tyxe.priors.LayerwiseNormalPrior(method="radford")
    prior.apply_(l)
    p = l._pyro_samples["weight"]
    assert p.base_dist.scale.allclose(torch.tensor(3 ** -0.5))


def test_layerwise_normal_xavier():
    l = PyroModule[nn.Linear](3, 2, bias=False)
    prior = tyxe.priors.LayerwiseNormalPrior(method="xavier")
    prior.apply_(l)
    p = l._pyro_samples["weight"]
    assert p.base_dist.scale.allclose(torch.tensor(0.8 ** 0.5))


def test_expose_all():
    net = PyroModule[nn.Sequential](PyroModule[nn.Linear](4, 3), PyroModule[nn.Linear](3, 2))
    tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=True).apply_(net)
    assert "weight" in net[0]._pyro_samples
    assert "bias" in net[0]._pyro_samples
    assert "weight" in net[1]._pyro_samples
    assert "bias" in net[1]._pyro_samples


def test_hide_all():
    net = PyroModule[nn.Sequential](PyroModule[nn.Linear](4, 3), PyroModule[nn.Linear](3, 2))
    tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, hide_all=True).apply_(net)
    assert "weight" in net[0]._pyro_params
    assert "bias" in net[0]._pyro_params
    assert "weight" in net[1]._pyro_params
    assert "bias" in net[1]._pyro_params


def test_expose_modules():
    net = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, expose_modules=[net[0]])
    tyxe.util.to_pyro_module_(net)
    prior.apply_(net)
    assert "weight" in net[0]._pyro_samples
    assert "bias" in net[0]._pyro_samples
    assert "weight" in net[1]._pyro_params
    assert "bias" in net[1]._pyro_params


def test_hide_modules():
    net = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, hide_modules=[net[0]])
    tyxe.util.to_pyro_module_(net)
    prior.apply_(net)
    assert "weight" in net[0]._pyro_params
    assert "bias" in net[0]._pyro_params
    assert "weight" in net[1]._pyro_samples
    assert "bias" in net[1]._pyro_samples


def test_expose_types():
    net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Linear(3, 2))
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, expose_module_types=(nn.Conv2d,))
    tyxe.util.to_pyro_module_(net)
    prior.apply_(net)
    assert "weight" in net[0]._pyro_samples
    assert "bias" in net[0]._pyro_samples
    assert "weight" in net[1]._pyro_params
    assert "bias" in net[1]._pyro_params


def test_hide_types():
    net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Linear(3, 2))
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, hide_module_types=(nn.Linear,))
    tyxe.util.to_pyro_module_(net)
    prior.apply_(net)
    assert "weight" in net[0]._pyro_samples
    assert "bias" in net[0]._pyro_samples
    assert "weight" in net[1]._pyro_params
    assert "bias" in net[1]._pyro_params


def test_expose_parameters():
    net = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, expose_parameters=["weight"])
    tyxe.util.to_pyro_module_(net)
    prior.apply_(net)
    assert "weight" in net[0]._pyro_samples
    assert "bias" in net[0]._pyro_params
    assert "weight" in net[1]._pyro_samples
    assert "bias" in net[1]._pyro_params


def test_hide_parameters():
    net = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, hide_parameters=["weight"])
    tyxe.util.to_pyro_module_(net)
    prior.apply_(net)
    assert "weight" in net[0]._pyro_params
    assert "bias" in net[0]._pyro_samples
    assert "weight" in net[1]._pyro_params
    assert "bias" in net[1]._pyro_samples


def test_expose():
    net = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, expose=["0.weight", "1.weight"])
    tyxe.util.to_pyro_module_(net)
    prior.apply_(net)
    assert "weight" in net[0]._pyro_samples
    assert "bias" in net[0]._pyro_params
    assert "weight" in net[1]._pyro_samples
    assert "bias" in net[1]._pyro_params


def test_hide():
    net = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, hide=["0.weight", "1.weight"])
    tyxe.util.to_pyro_module_(net)
    prior.apply_(net)
    assert "weight" in net[0]._pyro_params
    assert "bias" in net[0]._pyro_samples
    assert "weight" in net[1]._pyro_params
    assert "bias" in net[1]._pyro_samples
