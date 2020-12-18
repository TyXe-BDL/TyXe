import pytest

import torch
import torch.nn as nn

import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.nn.module import to_pyro_module_


import tyxe


def as_pyro_module(module):
    to_pyro_module_(module, recurse=True)
    for m in module.modules():
        for n, p in list(m.named_parameters(recurse=False)):
            setattr(m, n, PyroSample(dist.Normal(torch.zeros_like(p), torch.ones_like(p)).to_event()))
    return module


@pytest.mark.parametrize("reparameterization_ctxt", [tyxe.poutine.local_reparameterization, tyxe.poutine.flipout])
def test_different_outputs(reparameterization_ctxt):
    l = as_pyro_module(nn.Linear(3, 2))
    x = torch.randn(1, 3).repeat(128, 1)
    with reparameterization_ctxt():
        a = l(x)

    # compare the pairwise outputs by broadcasting, fail if any distinct pairs are equal
    assert not a.unsqueeze(-2).eq(a).all(-1).tril(diagonal=-1).any().item()


@pytest.mark.parametrize("reparameterization_ctxt", [tyxe.poutine.local_reparameterization, tyxe.poutine.flipout])
def test_mean_std(reparameterization_ctxt):
    d = 8
    n_samples = int(2 ** (d + 1))
    repeats = 1000

    x = torch.randn(d)

    l = PyroModule[nn.Linear](x.shape[0], 2)
    weight_mean = torch.randn_like(l.weight)
    weight_sd = torch.rand_like(l.weight)
    l.weight = PyroSample(dist.Normal(weight_mean, weight_sd).to_event())
    bias_mean = torch.randn_like(l.bias)
    bias_sd = torch.rand_like(l.bias)
    l.bias = PyroSample(dist.Normal(bias_mean, bias_sd).to_event())

    m = x @ weight_mean.t() + bias_mean
    s = torch.sqrt(x.pow(2) @ weight_sd.t().pow(2) + bias_sd.pow(2))

    x = x.unsqueeze(0).repeat(n_samples, 1)
    a = []
    for _ in range(repeats):
        with reparameterization_ctxt():
            a.append(l(x))
    a = torch.cat(a)

    assert torch.allclose(m, a.mean(0), atol=1e-2)
    assert torch.allclose(s, a.std(0), atol=1e-1)


def test_two_parameterizations_raises():
    l = as_pyro_module(nn.Linear(3,2))
    x = torch.randn(8, 3)
    with pytest.raises(ValueError):
        with tyxe.poutine.local_reparameterization(), tyxe.poutine.flipout():
            l(x)


def test_multiple_reparameterizers_compatible():
    net = as_pyro_module(nn.Sequential(
        nn.Conv2d(1, 2, 3),
        nn.Flatten(),
        nn.Linear(2, 3)
    ))
    x = torch.randn(1, 3, 3).repeat(8, 1, 1, 1)
    with tyxe.poutine.local_reparameterization(reparameterizable_functions="linear"),\
         tyxe.poutine.flipout(reparameterizable_functions="conv2d"):
        a = net(x)
    assert not a.unsqueeze(-2).eq(a).all(-1).tril(diagonal=-1).any().item()


def test_ignores_not_given_fn():
    l = as_pyro_module(nn.Linear(3, 2))
    x = torch.randn(1, 3).repeat(8, 1)
    with tyxe.poutine.local_reparameterization(reparameterizable_functions=["conv1d", "conv2d", "conv3d"]):
        a = l(x)
    assert a[0].eq(a).all().item()


def test_flipout_no_batch_dim():
    l = as_pyro_module(nn.Linear(3, 2))
    with tyxe.poutine.flipout():
        assert l(torch.randn(3)).shape == (2,)


def test_flipout_multi_batch_dim():
    l = as_pyro_module(nn.Linear(3, 2))
    with tyxe.poutine.flipout():
        assert l(torch.randn(5, 4, 3)).shape == (5, 4, 2)
