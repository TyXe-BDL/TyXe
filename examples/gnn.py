"""Bayesian Graph Neural Net, based on the DGL tutorial at https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
Also calculates expected calibration error as a metric """
import argparse
from functools import partial

import dgl
import dgl.function as fn
from dgl.data import citation_graph as citegrh

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist


import tyxe


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(1433, 16)
        self.layer2 = GCNLayer(16, 7)

    def forward(self, g, features):
        x = torch.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = dgl.from_networkx(data.graph)
    return g, features, labels, train_mask, test_mask


def calc_ece(probs, labels, num_bins):
    maxp, predictions = probs.max(-1, keepdims=True)
    boundaries = torch.linspace(0, 1, num_bins+1)
    lower_bound, upper_bound = boundaries[:-1], boundaries[1:]
    in_bin = maxp.ge(lower_bound).logical_and(maxp.lt(upper_bound)).float()
    bin_sizes = in_bin.sum(0)
    correct = predictions.eq(labels.unsqueeze(-1)).float()

    non_empty = bin_sizes.gt(0)
    accs = torch.where(non_empty, correct.mul(in_bin).sum(0) / bin_sizes, torch.zeros_like(bin_sizes))
    pred_probs = torch.where(non_empty, maxp.mul(in_bin).sum(0) / bin_sizes, torch.zeros_like(bin_sizes))
    bin_weight = bin_sizes / bin_sizes.sum()

    return accs.sub(pred_probs).abs().mul(bin_weight).sum()


def main(inference, lr, num_epochs, milestones):
    net = Net()

    g, features, labels, train_mask, test_mask = load_cora_data()
    # Add edges between each node and itself to preserve old node representations
    g.add_edges(g.nodes(), g.nodes())
    total_nodes = len(train_mask)
    training_nodes = train_mask.float().sum().item()

    prior_kwargs = {}
    if inference == "ml":
        # hide everything from the prior so that every nn.Parameter becomes a PyroParam
        prior_kwargs.update(expose_all=False, hide_all=True)
        # a guide is not needed in that case
        guide = None
    elif inference == "map":
        guide = pyro.infer.autoguide.AutoDelta
    elif inference == "mean-field":
        guide = partial(tyxe.guides.ParameterwiseDiagonalNormal, init_scale=1e-4, max_guide_scale=0.3,
                        init_loc_fn=tyxe.guides.SitewiseInitializer.from_net(net))
    else:
        raise RuntimeError("Unreachable")
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1), **prior_kwargs)

    # the dataset size needs to be set to the **total** number of nodes, since the pyro.plate receives all nodes
    # as a subsample, i.e. measures the batch size as equal to the total number of nodes, so we need to set the
    # dataset size accordingly to achieve the correct scaling of the log likelihood
    obs = tyxe.observation_models.Categorical(dataset_size=total_nodes)
    bnn = tyxe.VariationalBNN(net, prior, obs, guide)

    optim = torch.optim.Adam
    scheduler = pyro.optim.MultiStepLR({"optimizer": optim, "optim_args": {"lr": lr}, "milestones": milestones})
    # we only have one batch of data so it can go into a single-element list. BNN.fit assumes that the loader is an
    # iterator over two-element tuples, where the first element is a single element or tuple/list that is fed into the
    # NN and the second element is a tensor that contains the labels
    loader = [((g, features), labels)]

    def callback(b, i, e):
        errs = b.evaluate((g, features), labels, num_predictions=8, reduction="none")[0]
        acc = 1 - errs[test_mask].mean()
        ece = calc_ece(b.predict(g, features, num_predictions=8).softmax(-1)[test_mask], labels[test_mask], 10)
        print(f"Epoch {i+1:03d} | ELBO {e/training_nodes:03.4f} | Test Acc {100 * acc:.1f}% | ECE {100 * ece:.2f}%")
        scheduler.step()

    # the mask poutine is needed to only evaluate the log likelihood of the training nodes
    with tyxe.poutine.selective_mask(mask=train_mask, hide_all=False, expose=["observation_model.obs"]):
        bnn.fit(loader, scheduler, num_epochs, callback, num_particles=1)


if __name__ == "__main__":
    def list_of_ints(s):
        return list(map(int, s.split(",")))

    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", default="mean-field", choices=["ml", "map", "mean-field"])
    parser.add_argument("--lr", default=1e-1, type=float)
    parser.add_argument("--milestones", default=[100, 200, 300], type=list_of_ints)
    parser.add_argument("--num-epochs", default=400, type=int)
    main(**vars(parser.parse_args()))
