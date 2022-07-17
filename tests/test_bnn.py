from functools import partial

import torch
import torch.nn as nn
import torch.utils.data as data

import pyro
import pyro.distributions as dist
import pyro.infer.autoguide as ag
from pyro.infer.mcmc import HMC


import tyxe


def bayesian_regression(n, d, weight_precision, noise_precision):
    x = torch.randn(n, d)
    w = weight_precision ** -0.5 * torch.randn(d, 1)
    y = x @ w + noise_precision ** -0.5 * torch.randn(n, 1)

    posterior_precision = noise_precision * x.t().mm(x) + weight_precision * torch.eye(d)
    posterior_mean = torch.cholesky_solve(noise_precision * x.t().mm(y), torch.linalg.cholesky(posterior_precision))

    return x, y, w, posterior_precision, posterior_mean


def get_linear_bnn(n, d, wp, np, guide, variational=True):
    l = nn.Linear(d, 1, bias=False)
    prior = tyxe.priors.IIDPrior(dist.Normal(0, wp ** -0.5))
    likelihood = tyxe.likelihoods.HomoskedasticGaussian(n, precision=np)
    if variational:
        return tyxe.VariationalBNN(l, prior, likelihood, guide)
    else:
        return tyxe.MCMC_BNN(l, prior, likelihood, guide)

def test_diagonal_svi():
    torch.manual_seed(42)
    n, d, wp, np = 20, 2, 1, 100
    x, y, w, pp, pm = bayesian_regression(n, d, wp, np)
    bnn = get_linear_bnn(n, d, wp, np, partial(ag.AutoDiagonalNormal, init_scale=1e-2))

    loader = data.DataLoader(data.TensorDataset(x, y), n // 2, shuffle=True)

    optim = torch.optim.Adam
    sched = pyro.optim.StepLR({"optimizer": optim, "optim_args": {"lr": 1e-1}, "step_size": 100})
    bnn.fit(loader, sched, int(500), callback=lambda *args: sched.step())

    vm = pyro.get_param_store()["net_guide.loc"].data.squeeze()
    vp = pyro.get_param_store()["net_guide.scale"].data.squeeze()

    assert torch.allclose(vm, pm.squeeze(), atol=1e-2)
    assert torch.allclose(vp, pp.diagonal().sqrt().reciprocal(), atol=1e-2)


def test_multivariate_svi():
    torch.manual_seed(42)
    n, d, wp, np = 20, 2, 1, 100
    x, y, w, pp, pm = bayesian_regression(n, d, wp, np) # These are unchanged by the upgrade
    bnn = get_linear_bnn(n, d, wp, np, partial(ag.AutoMultivariateNormal, init_scale=1e-2))
    loader = data.DataLoader(data.TensorDataset(x, y), n // 2, shuffle=True)

    optim = torch.optim.Adam
    sched = pyro.optim.StepLR({"optimizer": optim, "optim_args": {"lr": 1e-1}, "step_size": 500})
    bnn.fit(loader, sched, 2500, num_particles=4, callback=lambda *args: sched.step())

    vm = pyro.get_param_store()["net_guide.loc"].data.squeeze()
    vsd = pyro.get_param_store()["net_guide.scale"].data
    vst = pyro.get_param_store()["net_guide.scale_tril"].data

    vs =  vst*vsd 

    assert torch.allclose(vm, pm.squeeze(), atol=0.01)

    cov_prec_mm = vs.mm(vs.t()).mm(pp)

    assert torch.allclose(cov_prec_mm, torch.eye(d), atol=0.05)

    site_names = tyxe.util.pyro_sample_sites(bnn.net) 
    assert "weight" in site_names

    samples = next(tyxe.util.named_pyro_samples(bnn.net))
    assert "weight" in samples


def test_hmc():
    torch.manual_seed(42)
    n, d, wp, np = 20, 2, 1, 100
    x, y, w, pp, pm = bayesian_regression(n, d, wp, np)
    bnn = get_linear_bnn(n, d, wp, np, partial(HMC, step_size=1e-2, num_steps=10, target_accept_prob=0.7),
                         variational=False)

    loader = data.DataLoader(data.TensorDataset(x, y), n // 2, shuffle=True)
    mcmc = bnn.fit(loader, num_samples=4000, warmup_steps=1000, disable_progbar=True).get_samples()
    w_mcmc = mcmc["net.weight"]


    w_mean = w_mcmc.mean(0)
    w_cov = w_mcmc.transpose(-2, -1).mul(w_mcmc).mean(0) - w_mean.t().mm(w_mean)

    assert torch.allclose(w_mean.squeeze(), pm.squeeze(), atol=1e-2)
    cov_prec_mm = w_cov @ pp
    assert torch.allclose(cov_prec_mm, torch.eye(d), atol=0.05)
