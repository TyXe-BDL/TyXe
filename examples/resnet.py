import argparse
import contextlib
import functools
import os

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision

import pyro
import pyro.distributions as dist
import pyro.infer.autoguide as ag


import tyxe


NORMALIZERS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
}


def make_loaders(dataset, root, train_batch_size, test_batch_size, use_cuda):
    train_img_transforms = [torchvision.transforms.RandomCrop(size=32, padding=4, padding_mode='reflect'),
                            torchvision.transforms.RandomHorizontalFlip()]
    test_img_transforms = []
    tensor_transforms = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(*NORMALIZERS[dataset])]

    dataset_fn = getattr(torchvision.datasets, dataset.upper())
    train_transform = torchvision.transforms.Compose(train_img_transforms + tensor_transforms)
    train_data = dataset_fn(root, train=True, transform=train_transform, download=True)
    train_loader = data.DataLoader(
        train_data, train_batch_size, pin_memory=use_cuda, num_workers=2 * int(use_cuda), shuffle=True)

    test_transform = torchvision.transforms.Compose(test_img_transforms + tensor_transforms)
    test_data = dataset_fn(root, train=False, transform=test_transform, download=True)
    test_loader = data.DataLoader(test_data, test_batch_size)

    return train_loader, test_loader


def make_net(dataset, architecture):
    net = getattr(torchvision.models, architecture)(pretrained=True)
    if dataset.startswith("cifar"):
        net.conv1 = nn.Conv2d(3, net.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        num_classes = 10 if dataset.endswith("10") else 100
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def main(dataset, architecture, inference, train_batch_size, test_batch_size, local_reparameterization,
         max_guide_scale, rank, root, seed):
    pyro.set_rng_seed(seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = make_loaders(dataset, root, train_batch_size, test_batch_size, use_cuda)
    net = make_net(dataset, architecture).to(device)

    observation_model = tyxe.observation_models.Categorical(len(train_loader.sampler))

    num_predictions = 10
    prior_kwargs = dict(expose_all=False, hide_module_types=(nn.BatchNorm2d,))
    if inference == "ml":
        num_predictions = 1
        guide = None
        prior_kwargs["hide_all"] = True
    elif inference == "map":
        num_predictions = 1
        guide = functools.partial(ag.AutoDelta, init_loc_fn=ezbnn.guides.SitewiseInitializer.from_net(net))
    elif inference == "mean-field":
        guide = functools.partial(tyxe.guides.ParameterwiseDiagonalNormal,
                                  init_loc_fn=tyxe.guides.SitewiseInitializer.from_net(net), init_scale=1e-4)
    elif inference.startswith("last-layer"):
        del prior_kwargs['hide_module_types']
        prior_kwargs["expose_modules"] = [net.fc]
        if inference == "last-layer-mean-field":
            guide = functools.partial(tyxe.guides.ParameterwiseDiagonalNormal,
                                      init_loc_fn=tyxe.guides.SitewiseInitializer.from_net(net), init_scale=1e-4)
        elif inference == "last-layer-full":
            guide = functools.partial(ag.AutoMultivariateNormal,
                                      init_loc_fn=tyxe.guides.SitewiseInitializer.from_net(net), init_scale=1e-4)
        elif inference == "last-layer-low-rank":
            guide = functools.partial(ag.AutoLowRankMultivariateNormal, rank=rank,
                                      init_loc_fn=tyxe.guides.SitewiseInitializer.from_net(net), init_scale=1e-4)
    else:
        raise RuntimeError("Unreachable")

    prior = tyxe.priors.IIDPrior(dist.Normal(torch.zeros(1, device=device), torch.ones(1, device=device)),
                                  **prior_kwargs)
    bnn = tyxe.SupervisedBNN(net, prior, observation_model, guide)

    fit_ctxt = tyxe.poutine.local_reparameterization if local_reparameterization else contextlib.nullcontext
    if max_guide_scale is not None:
        # the ClampParamMessenger doesn't work the way I thought it would because pyro calls .unconstrained() on the
        # parameters, so the return value can't be modified since that turns it into a tensor. Modifying the
        # constraint attribute of the param directly might be a better approach
        raise NotImplementedError

    optim = pyro.optim.Adam({"lr": 1e-3})

    def callback(b, i, avg_elbo):
        avg_err, avg_ll = 0., 0.
        for x, y in iter(test_loader):
            err, ll = b.evaluate(x.to(device), y.to(device), num_predictions=num_predictions)
            avg_err += err / len(test_loader.sampler)
            avg_ll += ll / len(test_loader.sampler)
        print(f"ELBO={avg_elbo}; test error={100 * avg_err:.2f}%; LL={avg_ll:.4f}")

    with fit_ctxt():
        bnn.fit(train_loader, optim, 100, callback=callback, device=device)


if __name__ == '__main__':
    resnets = [n for n in dir(torchvision.models)
               if (n.startswith("resnet") or n.startswith("wide_resnet")) and n[-1].isdigit()]

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--architecture", default="resnet18", choices=resnets)
    parser.add_argument("--inference", required=True, choices=
    ["ml", "map", "mean-field", "last-layer-mean-field", "last-layer-full", "last-layer-low-rank"])

    parser.add_argument("--train-batch-size", type=int, default=250)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--local-reparameterization", action="store_true")
    parser.add_argument("--max-guide-scale", type=float)
    parser.add_argument("--rank", type=int, default=10)

    parser.add_argument("--root", default=os.environ.get("DATASETS_PATH", "./data"))
    parser.add_argument("--seed", type=int, default=42)

    main(**vars((parser.parse_args())))
