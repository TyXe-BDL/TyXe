import pytest

import torch
import torch.distributions as dist

import pyro


from tyxe.likelihoods import *


@pytest.mark.parametrize("event_dim,shape,scale", [
    (0, (3, 2), 0.1),
    (1, (4, 3), 1.),
    (2, (5, 4, 3, 2), 3.)
])
def test_hom_gaussian_log_likelihood(event_dim, shape, scale):
    lik = HomoskedasticGaussian(event_dim=event_dim, dataset_size=-1, scale=scale)
    predictions = torch.randn(shape)
    targets = torch.randn(shape)
    log_probs = dist.Normal(predictions, scale).log_prob(targets)
    if event_dim != 0:
        log_probs = log_probs.flatten(-event_dim).sum(-1)
    assert torch.allclose(log_probs, lik.log_likelihood(predictions, targets))


@pytest.mark.parametrize("event_dim,shape", [
    (0, (3, 2)),
    (1, (4, 3)),
    (2, (5, 4, 3, 2))
])
def test_het_gaussian_log_likelihood(event_dim, shape):
    lik = HeteroskedasticGaussian(event_dim=event_dim, dataset_size=-1, positive_scale=True)
    pred_means = torch.randn(shape)
    pred_scales = torch.rand(shape)
    predictions = torch.cat((pred_means, pred_scales), dim=-1)
    targets = torch.randn(shape)
    log_probs = dist.Normal(pred_means, pred_scales).log_prob(targets)
    if event_dim != 0:
        log_probs = log_probs.flatten(-event_dim).sum(-1)
    assert torch.allclose(log_probs, lik.log_likelihood(predictions, targets))


@pytest.mark.parametrize("logits", [True, False])
@pytest.mark.parametrize("event_dim,shape", [
    (0, (3, 2)),
    (1, (4, 3)),
    (2, (5, 4, 3, 2))
])
def test_bernoulli_log_likelihood(event_dim, shape, logits):
    lik = Bernoulli(event_dim=event_dim, dataset_size=-1, logit_predictions=logits)
    targets = torch.randint(2, size=shape).float()
    if logits:
        predictions = torch.randn(shape)
        d = dist.Bernoulli(logits=predictions)
    else:
        predictions = torch.rand(shape)
        d = dist.Bernoulli(probs=predictions)
    log_probs = d.log_prob(targets)
    if event_dim != 0:
        log_probs = log_probs.flatten(-event_dim).sum(-1)
    assert torch.allclose(log_probs, lik.log_likelihood(predictions, targets))


@pytest.mark.parametrize("logits", [True, False])
@pytest.mark.parametrize("event_dim,shape", [
    (0, (3, 2, 5)),
    (1, (4, 3, 10)),
    (2, (5, 4, 3, 2, 7))
])
def test_categorical_log_likelihood(event_dim, shape, logits):
    lik = Categorical(event_dim=event_dim, dataset_size=-1, logit_predictions=logits)
    targets = torch.randint(shape[-1], size=shape[:-1])
    if logits:
        predictions = torch.randn(shape)
        d = dist.Categorical(logits=predictions)
    else:
        predictions = torch.randn(shape).softmax(-1)
        d = dist.Categorical(probs=predictions)
    log_probs = d.log_prob(targets)
    if event_dim != 0:
        log_probs = log_probs.flatten(-event_dim).sum(-1)
    assert torch.allclose(log_probs, lik.log_likelihood(predictions, targets))


@pytest.mark.parametrize("event_dim,shape", [
    (0, (3, 2)),
    (1, (4, 3)),
    (2, (5, 4, 3, 2))
])
def test_hom_gaussian_error(event_dim, shape):
    lik = HomoskedasticGaussian(event_dim=event_dim, dataset_size=-1, scale=1)
    predictions = torch.randn(shape)
    targets = torch.randn(shape)
    errors = (predictions - targets) ** 2
    if event_dim != 0:
        errors = errors.flatten(-event_dim).sum(-1)
    assert torch.allclose(errors, lik.error(predictions, targets))


@pytest.mark.parametrize("event_dim,shape", [
    (0, (3, 2)),
    (1, (4, 3)),
    (2, (5, 4, 3, 2))
])
def test_het_gaussian_error(event_dim, shape):
    lik = HeteroskedasticGaussian(event_dim=event_dim, dataset_size=-1, positive_scale=True)
    pred_means = torch.randn(shape)
    predictions = torch.cat((pred_means, torch.rand(shape)), dim=-1)
    targets = torch.randn(shape)
    errors = (pred_means - targets) ** 2
    if event_dim != 0:
        errors = errors.flatten(-event_dim).sum(-1)
    assert torch.allclose(errors, lik.error(predictions, targets))


@pytest.mark.parametrize("logits", [True, False])
@pytest.mark.parametrize("event_dim,shape", [
    (0, (3, 2)),
    (1, (4, 3)),
    (2, (5, 4, 3, 2))
])
def test_bernoulli_error(event_dim, shape, logits):
    lik = Bernoulli(event_dim=event_dim, dataset_size=-1, logit_predictions=logits)
    targets = torch.randint(2, size=shape).bool()
    if logits:
        predictions = torch.randn(shape)
        hard_predictions = predictions.gt(0)
    else:
        predictions = torch.rand(shape)
        hard_predictions = predictions.gt(0.5)
    errors = hard_predictions.ne(targets).float()
    if event_dim != 0:
        errors = errors.flatten(-event_dim).sum(-1)
    assert torch.allclose(errors, lik.error(predictions, targets.float()))


@pytest.mark.parametrize("logits", [True, False])
@pytest.mark.parametrize("event_dim,shape", [
    (0, (3, 2, 5)),
    (1, (4, 3, 10)),
    (2, (5, 4, 3, 2, 7))
])
def test_categorical_error(event_dim, shape, logits):
    lik = Categorical(event_dim=event_dim, dataset_size=-1, logit_predictions=logits)
    targets = torch.randint(shape[-1], size=shape[:-1])
    if logits:
        predictions = torch.randn(shape)
    else:
        predictions = torch.randn(shape).softmax(-1)
    hard_predictions = predictions.argmax(-1)
    errors = hard_predictions.ne(targets).float()
    if event_dim != 0:
        errors = errors.flatten(-event_dim).sum(-1)
    assert torch.allclose(errors, lik.error(predictions, targets.float()))


@pytest.mark.parametrize("agg_dim,scale", [(0, 0.1), (1, 1.), (2, 3.)])
def test_hom_gaussian_aggregate(agg_dim, scale):
    shape = (5, 4, 3)
    lik = HomoskedasticGaussian(event_dim=1, dataset_size=-1, scale=scale)
    predictions = torch.randn(shape)
    means, scales = lik.aggregate_predictions(predictions, agg_dim)
    true_means = predictions.mean(agg_dim)
    true_scale = predictions.var(agg_dim).add(scale ** 2).sqrt()

    assert torch.allclose(means, true_means)
    assert torch.allclose(scales, true_scale)


@pytest.mark.parametrize("agg_dim", [0, 1, 2])
def test_het_gaussian_aggregate(agg_dim):
    shape = (5, 4, 3)
    lik = HeteroskedasticGaussian(event_dim=1, dataset_size=-1, positive_scale=True)
    pred_means = torch.randn(shape)
    pred_scales = torch.rand(shape)
    pred_precisions = pred_scales.pow(-2)
    predictions = torch.cat((pred_means, pred_scales), dim=-1)
    means, scales = lik.aggregate_predictions(predictions, agg_dim).chunk(2, dim=-1)
    true_means = pred_means.mul(pred_precisions).sum(agg_dim) / pred_precisions.sum(agg_dim)
    true_scale = pred_means.var(agg_dim).add(pred_scales.pow(2).mean(agg_dim)).sqrt()

    assert torch.allclose(means, true_means)
    assert torch.allclose(scales, true_scale)


@pytest.mark.parametrize("logits", [True, False])
@pytest.mark.parametrize("agg_dim", [0, 1, 2])
def test_bernoulli_aggregate(agg_dim, logits):
    shape = (5, 4, 3)
    lik = Bernoulli(event_dim=1, dataset_size=-1, logit_predictions=logits)
    if logits:
        predictions = torch.randn(shape)
        avg_probs = predictions.sigmoid().mean(agg_dim)
        agg_predictions = avg_probs.log() - avg_probs.mul(-1).log1p()
    else:
        predictions = torch.rand(shape)
        agg_predictions = predictions.mean(agg_dim)

    assert torch.allclose(agg_predictions, lik.aggregate_predictions(predictions, agg_dim))


@pytest.mark.parametrize("logits", [True, False])
@pytest.mark.parametrize("agg_dim", [0, 1, 2])
def test_categorical_aggregate(agg_dim, logits):
    shape = (5, 4, 3, 10)
    lik = Categorical(event_dim=1, dataset_size=-1, logit_predictions=logits)
    if logits:
        predictions = torch.randn(shape)
        agg_predictions = predictions.softmax(-1).mean(agg_dim).log()
    else:
        predictions = torch.randn(shape).softmax(-1)
        agg_predictions = predictions.mean(agg_dim)
    assert torch.allclose(agg_predictions, lik.aggregate_predictions(predictions, agg_dim))


def test_forward_batch():
    shape = (4, 3)
    lik = Bernoulli(event_dim=1, dataset_size=10, logit_predictions=True)
    predictions = torch.randn(shape)
    tr = pyro.poutine.trace(lik).get_trace(predictions)
    assert tr.nodes["data"]["scale"] == 2.5


def test_forward_single():
    shape = (3,)
    lik = Bernoulli(event_dim=1, dataset_size=10, logit_predictions=True)
    predictions = torch.randn(shape)
    tr = pyro.poutine.trace(lik).get_trace(predictions)
    assert tr.nodes["data"]["scale"] == 10


def test_hom_gaussian_dist(): pass
