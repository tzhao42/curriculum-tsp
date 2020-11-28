"""Defines the main task for the TSP

The TSP is defined by the following traits:
    1. Each city in the list must be visited once and only once
    2. The salesman must return to the original node at the end of the tour

Since the TSP doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in trainer.py to be None

"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from . import node_distrib
from .node_distrib import get_param_nodes


class _TSPStage:
    """Stage of curriculum for training on tsp task.

    This is a helper object to keep track of a few things.

    Attributes:
        num_tiles (int): number of tiles in this stage
        param (torch.Tensor): parameter describing the data distribution
        start (int): the epoch number at which this stage activates
        length (int): the number of epochs this stage runs for
    """

    def __init__(self, num_tiles, param, start, length):
        """Create TSPStage instance.

        Args:
            num_tiles (int): number of tiles in this stage
            param (torch.Tensor): parameter describing the data distribution
            start (int): the epoch number at which this stage activates
            length (int): the number of epochs this stage runs for
        """
        self.num_tiles = num_tiles
        self.param = param
        self.start = start
        self.length = length


class TSPCurriculum:
    """Curriculum for training on tsp task."""

    def __init__(
        self,
        num_nodes,
        train_size,
        val_size,
        num_processes,
        regen=False,
        debug=False,
    ):
        """Create TSP curriculum.

        Args:
            num_nodes (int): number of nodes per problem instance
            train_size (int): number of training points
            val_size (int): number of validation points
            num_processes (int): number of processes used to generate data
            regen (bool): whether to regenerate each dataset every epoch
            debug (bool): whether to run in debug (visualization) mode
        """
        self._num_nodes = num_nodes
        self._train_size = train_size
        self._val_size = val_size
        self._num_processes = num_processes
        self._regen = regen
        self._debug = debug

        # need to clean this up

        # training (initialization values must be reset)
        self._all_stages = list()
        self._stages = None  # a generator

        self._curr_stage = None
        self._curr_dataset = None

        self._curr_epoch = -1
        self._curr_len = 0

        self._finished = False  # need to figure out how to finish up

        # validation
        self._val_dataset = None

    def _generate_curr_dataset(self):
        """Generate current training dataset.

        Call this helper method to generate a dataset with the current stage.
        """
        self._curr_dataset = TSPDataset(
            num_nodes=self._num_nodes,
            num_samples=self._train_size,
            num_tiles=self._curr_stage.num_tiles,
            param=self._curr_stage.param,
            num_processes=self._num_processes,
        )

    def increment_epoch(self):
        """Increment the current epoch of the curriculum.

        Indicates that we have trained for an epoch.
        """
        # TODO: clean up

        assert not self._finished

        self._curr_epoch += 1

        if self._curr_epoch == self._curr_len:
            # can't increment anymore
            self._finished = True

        if (
            self._curr_epoch
            == self._curr_stage.start + self._curr_stage.length
            and not self._finished
        ):
            # load new stage and new dataset
            self._curr_stage = next(self._stages)
            if not self._regen:
                self._generate_curr_dataset()

    def get_dataset(self):
        """Get the training dataset of the current epoch."""
        assert not self._finished
        if self._regen:
            self._generate_curr_dataset()

        if self._debug:
            self.visualize_dataset(self._curr_dataset)
        return self._curr_dataset

    def get_val_dataset(self):
        """Get the validation dataset."""
        if self._debug:
            self.visualize_dataset(self._val_dataset, val=True)
        return self._val_dataset

    def add_stage(self, num_tiles, param, num_epochs):
        """Add a training stage to the curriculum.

        Args
            num_tiles (int): number of tiles
            param (torch.Tensor): parameter for distribution of nodes
            num_epochs (int): number of epochs to train on this distribution
        """
        assert not self._finished
        curr_stage = _TSPStage(num_tiles, param, self._curr_len, num_epochs)
        self._all_stages.append(curr_stage)
        self._curr_len += num_epochs

    def add_val(self, num_tiles, param):
        """Add a validation dataset.

        Args
            num_tiles (int): number of tiles
            param (torch.Tensor): parameter for distribution of nodes
        """
        assert not self._finished
        assert self._val_dataset is None

        self._val_dataset = TSPDataset(
            num_nodes=self._num_nodes,
            num_samples=self._val_size,
            num_tiles=num_tiles,
            param=param,
            num_processes=1,
        )

    def start(self):
        """Start up the curriculum after adding all training stages."""
        assert not self._finished
        assert self._val_dataset

        self._curr_epoch = 0
        self._stages = (stage for stage in self._all_stages)
        self._curr_stage = next(self._stages)
        self._generate_curr_dataset()

    def visualize_dataset(self, tspdataset, val=False):
        """Print a visualization of current dataset."""
        x = tspdataset.dataset[:, 0, :].flatten().numpy()
        y = tspdataset.dataset[:, 1, :].flatten().numpy()

        plt.clf()
        plt.scatter(x, y)
        if val:
            plt.title("Validation")
        else:
            plt.title(f"Epoch {self._curr_epoch}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.draw()
        plt.pause(1)
        plt.close()
        plt.clf()


class TSPDataset(Dataset):
    def __init__(
        self, num_nodes, num_samples, num_tiles, param, num_processes
    ):
        """Create TSP dataset.

        Args:
            num_nodes (int): number of nodes per problem instance
            num_samples (int): number of problem instances in dataset
            num_tiles (int): number of tiles to split [0,1]x[0,1] into
            param (torch.Tensor): parameter for distribution of nodes
            num_processes (int): number of processes to generate data with
        """
        super(TSPDataset, self).__init__()

        self.dataset = get_param_nodes(
            num_nodes, num_samples, num_tiles, param, num_processes
        )
        self.dynamic = torch.zeros(num_samples, 1, num_nodes)
        self.num_nodes = num_nodes
        self.size = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [])


# tsp has no update dynamic function (it points to None)


def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask


def reward(static, tour_indices):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1).detach()


def render(static, tour_indices, save_path):
    """Plots the found tours."""

    plt.close("all")

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(
        nrows=num_plots, ncols=num_plots, sharex="col", sharey="row"
    )

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        # End tour at the starting index
        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        # plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c="r", zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c="k", marker="*", zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=400)
