"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import os
import time
import pathlib
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from constants import BASE_DIR, LOG_DIR, DEVICE
from models import DRL4TSP, Encoder, StateCritic

from datasets import tsp, vrp
from datasets.tsp import TSPDataset
from datasets.vrp import VehicleRoutingDataset

from utils import tsp_or_tools
from utils.tsp_or_tools import get_batched_or_tsp


def parse_arguments():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Combinatorial Optimization")

    # their arguments

    # set random seed
    parser.add_argument("--seed", default=12345, type=int)

    # current_run
    parser.add_argument("--task", default="tsp")
    parser.add_argument("--test", action="store_true", default=False)

    # current task
    parser.add_argument("--train-size", default=1000000, type=int)
    parser.add_argument("--valid-size", default=1000, type=int)
    parser.add_argument("--nodes", dest="num_nodes", default=20, type=int)

    # model and training params
    parser.add_argument("--actor_lr", default=5e-4, type=float)
    parser.add_argument("--critic_lr", default=5e-4, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden", dest="hidden_size", default=128, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--layers", dest="num_layers", default=1, type=int)

    # parser.add_argument("--checkpoint", default=None)

    # our arguments
    parser.add_argument("--run-name", default="tsp", type=str)
    parser.add_argument("--proportions", nargs=3, default=None, type=float)
    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--log_dir", default=None)

    # debug flag: short circuits training cycle
    parser.add_argument("--debug", dest="debug", default=False, action="store_true")

    return parser.parse_args()


class Run:
    """Object controlling runs."""

    def __init__(self, task, data_distrib, nodes, run_name, load=None):
        """
        Initialize a run object. In order to load, task and nodes must be the
        same as in the recorded run, while data_distrib and run_name do not.
        This is to facilitate training on one distribution and testing on
        another.

        Args:
            task: string, "tsp" or "vrp"
            data_distrib: list of floats, latent vector representing
                distributionfor points in dataset, where each float is between
                0 and 1 and the sum is 1
            nodes: int, one of: [10, 20, 50, 100]
            run_name: string, describes current run
            load: string, name of a run to be loaded
        """
        # checking validity of input
        assert task in {"tsp", "vrp"}
        for p in data_distrib:
            assert isinstance(p, float) or isinstance(p, int)
            assert p >= 0 and p <= 1
        assert sum(data_distrib) == 1
        assert nodes in {10, 20, 50, 100}
        assert isinstance(run_name, str)

        if load:
            # parse directories from load name
            load_task, load_nodes = load.split("-")[0:2]
            assert task == load_task
            assert nodes == load_nodes

        # generating current datetime
        # this is a bit hacky but I don't have time to make it better
        c_datetime = datetime.datetime.now()
        c_year = curr_datetime.year
        c_month = "{:02}".format(curr_datetime.month)
        c_day = "{:02}".format(curr_datetime.day)
        c_hour = "{:02}".format(curr_datetime.hour)
        c_min = "{:02}".format(curr_datetime.minute)
        c_sec = "{:02}".format(curr_datetime.second)
        curr_time_str = f"{c_year}{c_month}{c_day}T{c_hour}{c_min}{c_sec}"

        # getting parentdir
        parentdir = self._get_parentdir(task, nodes)

        # generating name
        name = load if load else f"{task}-{nodes}-{run_name}-{curr_time_str}"

        # initializing attributes
        self.task = task
        self.data_distrib = data_distrib
        self.nodes = nodes
        self.dir = os.path.join(parentdir, name)
        self.validate_dir = os.path.join(self.dir, "validate")
        self.checkpoint_dir = os.path.join(self.dir, "checkpoints")
        self.actor_path = os.path.join(self.dir, "actor.pt")
        self.critic_path = os.path.join(self.dir, "critic.pt")
        self._logfile = os.path.join(self.dir, "log.log")

        # creating/checking directories, if necessary
        if load:
            # things must exist
            assert os.path.exists(self.dir)
            assert os.path.exists(self.validate_dir)
            assert os.path.exists(self.checkpoint_dir)
            assert os.path.exists(self.actor_path)
            assert os.path.exists(self.critic_path)
            assert os.path.exists(self._logfile)
        else:
            # things must not yet exist, and need to be created
            assert not os.path.exists(self.dir)
            assert not os.path.exists(self.validate_dir)
            assert not os.path.exists(self.checkpoint_dir)
            assert not os.path.exists(self.actor_path)
            assert not os.path.exists(self.critic_path)
            assert not os.path.exists(self._logfile)
            os.makedirs(self.dir)
            os.makedirs(self.validate_dir)
            os.makedirs(self.checkpoint_dir)

        # initializing log file
        self.log(f"Run {curr_time_str}\n")
        self.log(f"Current device: {device}\n")
        self.log(f"Current data distribution: {data_distrib}\n")

    def log(self, message):
        """Writes a line to the log file."""
        with open(self._logfile, "a+") as f:
            f.write(message)

    def _get_parentdir(self, task, nodes):
        """Return parentdir path, throw error if it does not exist."""
        parentdir_name = f"{task}-{nodes}"
        parentdir = os.path.join(LOG_DIR, parentdir_name)
        assert os.path.exists(parentdir), f"Parent directory {parentdir} does not exist"
        return parentdir


def test(
    data_loader, actor, reward_fn, render_fn=None, num_plot=5, logger=None, debug=False
):
    """Compare performance of model and google OR tools"""

    actor.eval()

    optimality_gaps = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        # compute optimality gap
        model_tour_lengths = reward_fn(static, tour_indices)
        optimal_tour_lengths = get_batched_or_tsp(static)  # not actually exact solution
        curr_opt_gaps = model_tour_lengths.cpu() / optimal_tour_lengths.cpu()
        mean_opt_gap = curr_opt_gaps.mean().item()
        optimality_gaps.append(mean_opt_gap)

        if render_fn is not None and batch_idx < num_plot:
            name = "batch%d_%2.4f.png" % (batch_idx, mean_opt_gap)
            path = os.path.join(logger.validate_dir, name)
            render_fn(static, tour_indices, path)

        if debug:
            break

    actor.train()
    return np.mean(optimality_gaps)


def validate(
    data_loader, actor, reward_fn, render_fn=None, num_plot=5, logger=None, debug=False
):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = "batch%d_%2.4f.png" % (batch_idx, reward)
            path = os.path.join(logger.validate_dir, name)
            render_fn(static, tour_indices, path)

        if debug:
            break

    actor.train()
    return np.mean(rewards)


def train(
    actor,
    critic,
    task,
    num_nodes,
    train_data,
    valid_data,
    reward_fn,
    render_fn,
    batch_size,
    actor_lr,
    critic_lr,
    max_grad_norm,
    logger=None,
    **kwargs,
):
    """Constructs the main actor & critic networks, and performs all training."""

    save_dir = logger.dir
    checkpoint_dir = logger.checkpoint_dir

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = np.inf

    for epoch in range(20):

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):

            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            advantage = reward - critic_est
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            # if (batch_idx + 1) % 100 == 0:
            if batch_idx % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                logger.log(
                    "Epoch %d, Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs\n"
                    % (
                        epoch,
                        batch_idx,
                        len(train_loader),
                        mean_reward,
                        mean_loss,
                        times[-1],
                    )
                )

            if kwargs["debug"]:
                break

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, f"{epoch}")
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, "actor.pt")
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, "critic.pt")
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        mean_valid = validate(
            valid_loader,
            actor,
            reward_fn,
            render_fn,
            num_plot=5,
            logger=logger,
            debug=args.debug,
        )

        # Save best model parameters
        if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, "actor.pt")
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, "critic.pt")
            torch.save(critic.state_dict(), save_path)

        logger.log(
            "Epoch %d, Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs "
            "(%2.4fs / 100 batches)\n"
            % (
                epoch,
                mean_loss,
                mean_reward,
                mean_valid,
                time.time() - epoch_start,
                np.mean(times),
            )
        )

        if kwargs["debug"]:
            break


def train_tsp(args):
    """Main method for training model for tsp task."""

    # Goals from paper:
    # TSP20, 3.97
    # TSP50, 6.08
    # TSP100, 8.44

    # checking if debug
    if args.debug:
        args.train_size = 10
        args.valid_size = 10

    # initializing logger
    logger = Logger(args.task, args.num_nodes, args.run_name, args.log_dir)

    # creating datasets
    train_data = TSPDataset(
        args.num_nodes, args.train_size, args.seed, args.proportions
    )
    # train_data = TSPDataset(args.num_nodes, args.train_size, args.seed, proportions=[0,1.0,0])
    valid_data = TSPDataset(
        args.num_nodes, args.valid_size, args.seed + 1, args.proportions
    )
    # valid_data = TSPDataset(args.num_nodes, args.valid_size, args.seed + 1, proportions=[0,1.0,0])

    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 1  # dummy for compatibility

    update_fn = None

    actor = DRL4TSP(
        STATIC_SIZE,
        DYNAMIC_SIZE,
        args.hidden_size,
        update_fn,
        tsp.update_mask,
        args.num_layers,
        args.dropout,
    ).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    kwargs["train_data"] = train_data
    kwargs["valid_data"] = valid_data
    kwargs["reward_fn"] = tsp.reward
    kwargs["render_fn"] = tsp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, "actor.pt")
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, "critic.pt")
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, logger=logger, **kwargs)

    test_proportions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    test_names = ["uniform", "shifted", "adversary"]

    for test_id in range(0, 3):
        test_data = TSPDataset(
            args.num_nodes,
            20,  # args.valid_size # cannot do too many because too slow
            args.seed + 2,
            proportions=test_proportions[test_id],
        )
        test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
        out = test(
            test_loader,
            actor,
            tsp.reward,
            tsp.render,
            num_plot=5,
            logger=logger,
            debug=args.debug,
        )
        logger.log(f"Average optimality gap for {test_names[test_id]}: {out}\n")


def train_vrp(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 2  # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    train_data = VehicleRoutingDataset(
        args.train_size, args.num_nodes, max_load, MAX_DEMAND, args.seed
    )

    valid_data = VehicleRoutingDataset(
        args.valid_size, args.num_nodes, max_load, MAX_DEMAND, args.seed + 1
    )

    actor = DRL4TSP(
        STATIC_SIZE,
        DYNAMIC_SIZE,
        args.hidden_size,
        train_data.update_dynamic,
        train_data.update_mask,
        args.num_layers,
        args.dropout,
    ).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    kwargs["train_data"] = train_data
    kwargs["valid_data"] = valid_data
    kwargs["reward_fn"] = vrp.reward
    kwargs["render_fn"] = vrp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, "actor.pt")
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, "critic.pt")
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = VehicleRoutingDataset(
        args.valid_size, args.num_nodes, max_load, MAX_DEMAND, args.seed + 2
    )

    test_dir = "test"
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)

    print("Average tour length: ", out)


if __name__ == "__main__":

    # parsing args
    args = parse_arguments()

    # loading from log_dir for evaluation
    # sample log_dir: tsp-20-1.00-0.00-0.00-20201114T193737
    # args.log_dir = os.path.join(LOG_DIR, f"{args.task}-{args.")

    # print('NOTE: SETTTING CHECKPOINT: ')
    # args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    # print(args.checkpoint)

    if DEVICE == "cuda":
        with torch.cuda.device(args.device_id):
            if args.task == "tsp":
                train_tsp(args)
            elif args.task == "vrp":
                train_vrp(args)
            else:
                raise ValueError("Task <%s> not understood" % args.task)
