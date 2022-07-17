import argparse
import copy
import functools
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision.transforms as tf
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

import pyro
import pyro.distributions as dist


import tyxe


ROOT = os.environ.get("DATASETS_PATH", "./data")
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if USE_CUDA else torch.device("cpu")
C10_MEAN = (0.49139968, 0.48215841, 0.44653091)
C10_SD = (0.24703223, 0.24348513, 0.26158784)


def conv_3x3(c_in, c_out):
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)


class ConvNet(nn.Sequential):

    def __init__(self):
        super().__init__()
        self.add_module("Conv1_1", conv_3x3(3, 32))
        self.add_module("ReLU1_1", nn.ReLU(inplace=True))
        self.add_module("Conv1_2", conv_3x3(32, 32))
        self.add_module("ReLU1_2", nn.ReLU(inplace=True))
        self.add_module("MaxPool1", nn.MaxPool2d(2, stride=2))

        self.add_module("Conv2_1", conv_3x3(32, 64))
        self.add_module("ReLU2_1", nn.ReLU(inplace=True))
        self.add_module("Conv2_2", conv_3x3(64, 64))
        self.add_module("ReLU2_2", nn.ReLU(inplace=True))
        self.add_module("MaxPool2", nn.MaxPool2d(2, stride=2))

        self.add_module("Flatten", nn.Flatten())

        self.add_module("Linear", nn.Linear(64 * 8 * 8, 512))
        self.add_module("ReLU", nn.ReLU(inplace=True))

        self.add_module("Head", nn.Linear(512, 10))


class FCNet(nn.Sequential):

    def __init__(self):
        super().__init__()
        self.add_module("Linear", nn.Linear(784, 200))
        self.add_module("ReLU", nn.ReLU(inplace=True))
        self.add_module("Head", nn.Linear(200, 1))


def make_mnist_dataloaders(root, train_batch_size, test_batch_size):
    train_loaders = []
    test_loaders = []

    for train, loaders, bs in zip((True, False), (train_loaders, test_loaders), (train_batch_size, test_batch_size)):
        mnist = MNIST(os.path.join(root, "mnist"), train=train, download=True)
        x = mnist.data.flatten(1) / 255.
        y = mnist.targets
        for i in range(5):
            index = y.ge(i * 2) & y.lt((i + 1) * 2)
            loaders.append(data.DataLoader(data.TensorDataset(x[index], y[index].sub(2 * i).float().unsqueeze(-1)),
                                           bs, shuffle=True, pin_memory=USE_CUDA))

    return train_loaders, test_loaders


def make_cifar_dataloaders(root, train_batch_size, test_batch_size):
    train_loaders = []
    test_loaders = []

    c100_means = []
    c100_sds = []
    for train, loaders, bs in zip((True, False), (train_loaders, test_loaders), (train_batch_size, test_batch_size)):
        c10 = CIFAR10(
                os.path.join(root, "cifar10"),
                train=train,
                transform=tf.Compose([tf.ToTensor(),
                tf.Normalize(C10_MEAN, C10_SD)])
            )

            
        loaders.append(data.DataLoader(c10, bs, shuffle=train, pin_memory=USE_CUDA))

        c100 = CIFAR100(os.path.join(root, "cifar100"), train=train)
        unnormalized_data = torch.from_numpy(c100.data).permute(0, 3, 1, 2).div(255.)  # convert images to torch arrays
        targets = torch.tensor(c100.targets)

        for i in range(5):
            index = targets.ge(i * 10) & targets.lt((i + 1) * 10)

            unnormalized_data_i = unnormalized_data[index]
            if train:
                c100_means.append(unnormalized_data_i.mean((0, 2, 3), keepdims=True))
                c100_sds.append(unnormalized_data_i.std((0, 2, 3), keepdims=True))
            normalized_data_i = (unnormalized_data_i - c100_means[i]) / c100_sds[i]
            targets_i = targets[index] - i * 10

            dataset_i = data.TensorDataset(normalized_data_i, targets_i)
            loaders.append(data.DataLoader(dataset_i, bs, shuffle=train, pin_memory=USE_CUDA))

    return train_loaders, test_loaders


def main(root, dataset, inference, num_epochs=0):
    train_batch_size = 250
    test_batch_size = 1000

    if dataset == "cifar":
        net = ConvNet()
        obs = tyxe.likelihoods.Categorical(-1)
        train_loaders, test_loaders = make_cifar_dataloaders(root, train_batch_size, test_batch_size)
        num_epochs = 60 if not num_epochs else num_epochs
    elif dataset == "mnist":
        net = FCNet()
        obs = tyxe.likelihoods.Bernoulli(-1, event_dim=1)
        train_loaders, test_loaders = make_mnist_dataloaders(root, train_batch_size, test_batch_size)
        num_epochs = 600 if not num_epochs else num_epochs
    else:
        raise RuntimeError("Unreachable")

    net.to(DEVICE)
    if inference == "mean-field":
        prior = tyxe.priors.IIDPrior(dist.Normal(torch.tensor(0., device=DEVICE), torch.tensor(1., device=DEVICE)),
                                     expose_all=False, hide_modules=[net.Head])
        guide = functools.partial(
            tyxe.guides.AutoNormal,  
            init_scale=1e-4,  
            init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(net, prefix="net")
        )
        test_samples = 8
    elif inference == "ml":
        prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), expose_all=False, hide_all=True)
        guide = None
    else:
        raise RuntimeError("Unreachable")
    bnn = tyxe.VariationalBNN(net, prior, obs, guide)
    n_tasks = len(train_loaders)
    test_errors = torch.ones(n_tasks, n_tasks)

    head_state_dicts = []
    init_head_sd = copy.deepcopy(net.Head.state_dict())
    for i, train_loader in enumerate(train_loaders, 1):
        elbos = []
        net.Head.load_state_dict(init_head_sd)

        pbar = tqdm(total=num_epochs, unit="Epochs", postfix=f"Task {i}")

        def callback(_i, _ii, e):
            elbos.append(e / len(train_loader.sampler))
            pbar.update()

        obs.dataset_size = len(train_loader.sampler)
        optim = pyro.optim.Adam({"lr": 1e-3})
        with tyxe.poutine.local_reparameterization():
            bnn.fit(train_loader, optim, num_epochs, device=DEVICE, callback=callback)

        pbar.close()

        head_state_dicts.append(copy.deepcopy(net.Head.state_dict()))
        for j, (test_loader, head_params) in enumerate(zip(test_loaders, head_state_dicts)):
            net.Head.load_state_dict(head_params)
            err = sum(bnn.evaluate(x.to(DEVICE), y.to(DEVICE), num_predictions=8)[0] for x, y in test_loader)
            test_errors[i-1, j] = err / len(test_loader.sampler)

        print("\t".join(["Error"] + [f"Task {j}" for j in range(1, i+1)]))
        print("\t" + "\t".join([f"{100 * e:.2f}%" for e in test_errors[i-1, :i]]))

        if inference == "mean-field":
            site_names = tyxe.util.pyro_sample_sites(bnn)
            bnn.update_prior(tyxe.priors.DictPrior(bnn.net_guide.get_detached_distributions(site_names)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    parser.add_argument("--inference", choices=["mean-field", "ml"], required=True)
    parser.add_argument("--num-epochs", default=0, required=False, type=int)
    main(**vars(parser.parse_args()))
