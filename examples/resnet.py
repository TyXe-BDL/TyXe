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
    "cifar10": ((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    "cifar100": ((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
    "svhn": ((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
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

    ood_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(*NORMALIZERS["svhn"])])
    ood_data = torchvision.datasets.SVHN(root, split="test", transform=ood_transform, download=True)
    ood_loader = data.DataLoader(ood_data, test_batch_size)

    return train_loader, test_loader, ood_loader


def make_net(dataset, architecture):
    net = getattr(torchvision.models, architecture)(pretrained=True)
    if dataset.startswith("cifar"):
        net.conv1 = nn.Conv2d(3, net.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        num_classes = 10 if dataset.endswith("10") else 100
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def main(dataset, architecture, inference, train_batch_size, test_batch_size, local_reparameterization, flipout,
         num_epochs, test_samples, max_guide_scale, rank, root, seed, output_dir, pretrained_weights, scale_only, lr, milestones, gamma):
    pyro.set_rng_seed(seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader, ood_loader = make_loaders(dataset, root, train_batch_size, test_batch_size, use_cuda)
    net = make_net(dataset, architecture).to(device)
    if pretrained_weights is not None:
        sd = torch.load(pretrained_weights, map_location=device)
        net.load_state_dict(sd)

    observation_model = tyxe.likelihoods.Categorical(len(train_loader.sampler))

    prior_kwargs = dict(expose_all=False, hide_module_types=(nn.BatchNorm2d,))
    if inference == "ml":
        test_samples = 1
        guide = None
        prior_kwargs["hide_all"] = True
    elif inference == "map":
        test_samples = 1
        guide = functools.partial(ag.AutoDelta, init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(net, prefix="net"))
    elif inference == "mean-field":
        guide = functools.partial(tyxe.guides.AutoNormal,
                                  init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(net, prefix="net"), init_scale=1e-4,
                                  max_guide_scale=max_guide_scale, train_loc=not scale_only)
    elif inference.startswith("last-layer"):
        if pretrained_weights is None:
            raise ValueError("Asked to do last-layer inference, but no pre-trained weights were provided.")
        # turning parameters except for last layer in buffers to avoid training them
        # this might be avoidable via poutine.block
        for module in net.modules():
            if module is not net.fc:
                for param_name, param in list(module.named_parameters(recurse=False)):
                    delattr(module, param_name)
                    module.register_buffer(param_name, param.detach().data)
        del prior_kwargs['hide_module_types']
        prior_kwargs["expose_modules"] = [net.fc]
        if inference == "last-layer-mean-field":
            guide = functools.partial(tyxe.guides.AutoNormal,
                                      init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(net, prefix="net"), init_scale=1e-4)
        elif inference == "last-layer-full":
            guide = functools.partial(ag.AutoMultivariateNormal,
                                      init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(net, prefix="net"), init_scale=1e-4)
        elif inference == "last-layer-low-rank":
            guide = functools.partial(ag.AutoLowRankMultivariateNormal, rank=rank,
                                      init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(net, prefix="net"), init_scale=1e-4)
        else:
            raise RuntimeError("Unreachable")
    else:
        raise RuntimeError("Unreachable")

    prior = tyxe.priors.IIDPrior(dist.Normal(torch.zeros(1, device=device), torch.ones(1, device=device)),
                                 **prior_kwargs)
    bnn = tyxe.VariationalBNN(net, prior, observation_model, guide)

    if local_reparameterization:
        if flipout:
            raise RuntimeError("Can't use both local reparameterization and flipout, pick one.")
        fit_ctxt = tyxe.poutine.local_reparameterization
    elif flipout:
        fit_ctxt = tyxe.poutine.flipout
    else:
        fit_ctxt = contextlib.nullcontext

    if milestones is None:
        optim = pyro.optim.Adam({"lr": lr})
    else:
        optimizer = torch.optim.Adam
        optim = pyro.optim.MultiStepLR({"optimizer": optimizer, "optim_args": {"lr": lr}, "milestones": milestones, "gamma": gamma})

    def callback(b, i, avg_elbo):
        avg_err, avg_ll = 0., 0.
        b.eval()
        for x, y in iter(test_loader):
            err, ll = b.evaluate(x.to(device), y.to(device), num_predictions=test_samples)
            avg_err += err / len(test_loader.sampler)
            avg_ll += ll / len(test_loader.sampler)
        print(f"ELBO={avg_elbo}; test error={100 * avg_err:.2f}%; LL={avg_ll:.4f}")
        b.train()

    with fit_ctxt():
        bnn.fit(train_loader, optim, num_epochs, callback=callback, device=device)

    if output_dir is not None:
        pyro.get_param_store().save(os.path.join(output_dir, "param_store.pt"))
        torch.save(bnn.state_dict(), os.path.join(output_dir, "state_dict.pt"))

        test_predictions = torch.cat([bnn.predict(x.to(device), num_predictions=test_samples)
                                      for x, _ in iter(test_loader)])
        torch.save(test_predictions.detach().cpu(), os.path.join(output_dir, "test_predictions.pt"))

        ood_predictions = torch.cat([bnn.predict(x.to(device), num_predictions=test_samples)
                                     for x, _ in iter(ood_loader)])
        torch.save(ood_predictions.detach().cpu(), os.path.join(output_dir, "ood_predictions.pt"))


if __name__ == '__main__':
    resnets = [n for n in dir(torchvision.models)
               if (n.startswith("resnet") or n.startswith("wide_resnet")) and n[-1].isdigit()]

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--architecture", default="resnet18", choices=resnets)
    parser.add_argument("--inference", required=True, choices=[
        "ml", "map", "mean-field", "last-layer-mean-field", "last-layer-full", "last-layer-low-rank"])

    parser.add_argument("--train-batch-size", type=int, default=100)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=20)
    parser.add_argument("--local-reparameterization", action="store_true")
    parser.add_argument("--flipout", action="store_true")
    parser.add_argument("--max-guide-scale", type=float)
    parser.add_argument("--rank", type=int, default=10)

    parser.add_argument("--root", default=os.environ.get("DATASETS_PATH", "./data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir")
    parser.add_argument("--pretrained-weights")
    parser.add_argument("--scale-only", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--milestones", type=lambda s: list(map(int, s.split(","))))
    parser.add_argument("--gamma", type=float, default=0.1)

    main(**vars((parser.parse_args())))
