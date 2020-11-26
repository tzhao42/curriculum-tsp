"""Defines the main trainer model for combinatorial problems

Each task must define the following functions:
* mask_fn: can be None
* update_fn: can be None
* reward_fn: specifies the quality of found solutions
* render_fn: Specifies how to plot found solutions. Can be None
"""

import argparse
import datetime
import os
import pathlib
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from constants import (
    DEBUG,
    DEVICE,
    LOG_DIR,
    NUM_TILES,
    ORTOOLS_TSP_TIMEOUT,
    STATIC_SIZE,
    TSP_DYNAMIC_SIZE,
    VRP_DYNAMIC_SIZE,
    VRP_LOAD_DICT,
    VRP_MAX_DEMAND,
)
from curriculums import get_indexed_curriculum
from models import DRL4TSP, Encoder, StateCritic
from tasks import node_distrib, tsp, vrp
from tasks.node_distrib import (
    get_down_line_param,
    get_uniform_param,
    get_up_line_param,
)
from tasks.tsp import TSPCurriculum, TSPDataset, update_mask
from tasks.vrp import VehicleRoutingDataset, update_dynamic, update_mask
from torch.utils.data import DataLoader
from utils import tsp_or_tools
from utils.tsp_or_tools import get_batched_or_tsp


def parse_arguments():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Combinatorial Optimization")

    # random seed
    parser.add_argument("--seed", default=12345, type=int)

    # current run and io
    parser.add_argument("--mode", default="train")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--load", default=None)

    # current task/dataset
    parser.add_argument("--task", default="tsp")
    parser.add_argument("--train-size", default=1000000, type=int)
    parser.add_argument("--val-size", default=1000, type=int)
    parser.add_argument("--num-nodes", default=20, type=int)
    parser.add_argument("--curriculum", default=0, type=int)

    # model params
    parser.add_argument("--hidden-size", default=128, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--num-layers", default=1, type=int)

    # training/testing params
    parser.add_argument("--actor-lr", default=5e-4, type=float)
    parser.add_argument("--critic-lr", default=5e-4, type=float)
    parser.add_argument("--max-grad-norm", default=2.0, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=20, type=int)

    # debug flag: short circuits training cycle
    parser.add_argument(
        "--debug", dest="debug", default=False, action="store_true"
    )

    return parser.parse_args()


def check_args_valid(args):
    """Check whether a set of args are valid."""
    assert isinstance(args.seed, int)
    assert args.mode in {"train", "test", "all"}
    assert isinstance(args.run_name, str)
    assert isinstance(args.device_id, int)
    if DEVICE == "cpu":
        assert args.device_id == 0
    elif DEVICE == "cuda":
        assert args.device_id + 1 <= torch.cuda.device_count()
    if args.load:
        # parse task and nodes from load name
        load_task, load_nodes = args.load.split("-")[0:2]
        assert args.task == load_task
        assert args.num_nodes == int(load_nodes)
    assert args.task in {"tsp", "vrp"}
    assert isinstance(args.train_size, int)
    assert isinstance(args.val_size, int)
    assert isinstance(args.num_nodes, int)
    assert isinstance(args.curriculum, int)
    assert args.curriculum >= 0
    assert isinstance(args.hidden_size, int)
    assert isinstance(args.dropout, float)
    assert isinstance(args.num_layers, int)
    assert isinstance(args.actor_lr, float)
    assert isinstance(args.critic_lr, float)
    assert isinstance(args.max_grad_norm, float)
    assert isinstance(args.batch_size, int)
    assert isinstance(args.epochs, int)
    assert isinstance(args.debug, bool)


class RunIO:
    """Object controlling io in runs.

    Attributes
        dir: log directory for this run
        validate_dir: validation directory for this run
        checkpoint_dir: checkpoint directory for this run
        actor_path: path to saved best actor
        critic_path: path to saved best critic
    """

    def __init__(self, task, nodes, run_name, load=None):
        """Initialize a run object.

        In order to load, task and nodes must be the same as in the recorded
        run, while run_name does not (it is ignored). This is checked in the
        check_args_valid function.

        (Yes this is over-engineered.)

        Args:
            task: string, "tsp" or "vrp"
            nodes: int, one of: [10, 20, 50, 100]
            run_name: string, describes current run
            load: string, name of a run to be loaded (i.e., tsp-20-epicmodel)
        """
        # generating current datetime
        # this is a bit hacky but I don't have time to make it better
        curr_datetime = datetime.datetime.now()
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
        self.log(f"\nRun {run_name} {curr_time_str}\n")
        self.log(f"Current device: {DEVICE}\n")
        # self.log(f"Current data distribution: {data_distrib}\n")

    def log(self, message):
        """Writes a line to the log file."""
        with open(self._logfile, "a+") as f:
            f.write(message)

    def _get_parentdir(self, task, nodes):
        """Return parentdir path, throw error if it does not exist."""
        parentdir_name = f"{task}-{nodes}"
        parentdir = os.path.join(LOG_DIR, parentdir_name)
        assert os.path.exists(
            parentdir
        ), f"Parent directory {parentdir} does not exist"
        return parentdir


def test_tsp(
    data_loader, actor, reward_fn, render_fn=None, num_plot=5, run_io=None
):
    """Compare performance of model and google OR tools."""

    actor.eval()

    cum_tour_length = 0
    cum_opt_gap = 0

    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(DEVICE)
        dynamic = dynamic.to(DEVICE)
        x0 = x0.to(DEVICE) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        # compute optimality gap
        model_tour_lengths = reward_fn(static, tour_indices)
        optimal_tour_lengths = get_batched_or_tsp(static, ORTOOLS_TSP_TIMEOUT)
        curr_opt_gaps = model_tour_lengths.cpu() / optimal_tour_lengths.cpu()

        # increment cumulative values
        cum_tour_length += model_tour_lengths.cpu().sum().item()
        cum_opt_gap += curr_opt_gaps.sum().item()

        mean_opt_gap = curr_opt_gaps.mean().item()
        if render_fn is not None and batch_idx < num_plot:
            name = "batch%d_%2.4f.png" % (batch_idx, mean_opt_gap)
            path = os.path.join(run_io.validate_dir, name)
            render_fn(static, tour_indices, path)

    # calculate and log results
    avg_tour_length = cum_tour_length / len(data_loader)
    avg_opt_gap = cum_opt_gap / len(data_loader)
    run_io.log(f"Average tour length: {avg_tour_length}\n")
    run_io.log(f"Average optimality gap: {avg_opt_gap}\n")


def validate(
    data_loader, actor, reward_fn, render_fn=None, num_plot=5, run_io=None
):
    """Used to monitor progress on a validation set & optionally plot."""

    actor.eval()

    cum_reward = 0

    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(DEVICE)
        dynamic = dynamic.to(DEVICE)
        x0 = x0.to(DEVICE) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward_tensor = reward_fn(static, tour_indices)
        reward = reward_tensor.cpu().mean().item()
        cum_reward += reward_tensor.cpu().sum().item()

        if render_fn is not None and batch_idx < num_plot:
            name = "batch%d_%2.4f.png" % (batch_idx, reward)
            path = os.path.join(run_io.validate_dir, name)
            render_fn(static, tour_indices, path)

    avg_reward = cum_reward / len(data_loader)

    return avg_reward


def train_single_epoch(
    actor,
    critic,
    train_loader,
    reward_fn,
    actor_opt,
    critic_opt,
    max_grad_norm,
    epoch,
    run_io,
):
    actor.train()
    critic.train()

    # Not sure why critic_rewards is being kept track of, but we'll keep that
    # the same for the time being
    times, losses, rewards, critic_rewards = [], [], [], []

    start = time.time()

    for batch_idx, batch in enumerate(train_loader):

        # getting batch
        static, dynamic, x0 = batch

        # sending batch to device
        static = static.to(DEVICE)
        dynamic = dynamic.to(DEVICE)
        x0 = x0.to(DEVICE) if len(x0) > 0 else None

        # Full forward pass through the dataset
        tour_indices, tour_logp = actor(static, dynamic, x0)

        # Sum the log probabilities for each city in the tour
        reward = reward_fn(static, tour_indices)

        # Query the critic for an estimate of the reward
        critic_est = critic(static, dynamic).view(-1)

        # Loss for each model
        advantage = reward - critic_est
        actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
        critic_loss = torch.mean(advantage ** 2)

        # optimize actor
        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        actor_opt.step()

        # optimize critic
        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        critic_opt.step()

        # keeping track of values for logging
        critic_rewards.append(torch.mean(critic_est.detach()).item())
        rewards.append(torch.mean(reward.detach()).item())
        losses.append(torch.mean(actor_loss.detach()).item())

        # Logging training progress
        if batch_idx % 100 == 0:
            end = time.time()
            times.append(end - start)
            start = end

            mean_loss = np.mean(losses[-100:])
            mean_reward = np.mean(rewards[-100:])

            run_io.log(
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

    return times, losses, rewards


def train_curriculum(
    actor,
    critic,
    curriculum,
    reward_fn,
    render_fn,
    epochs,
    batch_size,
    actor_lr,
    critic_lr,
    max_grad_norm,
    run_io,
):
    """Constructs the main actor & critic networks, and performs training."""
    # val loader
    val_loader = DataLoader(
        curriculum.get_val_dataset(), batch_size, shuffle=False, num_workers=0
    )

    # optimizers
    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    best_params = None
    best_reward = np.inf

    for epoch in range(epochs):
        # keeping track of start time
        epoch_start = time.time()

        # get current train loader
        curr_train_data = curriculum.get_dataset()
        curr_train_loader = DataLoader(
            curr_train_data, batch_size, shuffle=True, num_workers=0
        )

        # training for an epoch
        times, losses, rewards = train_single_epoch(
            actor,
            critic,
            curr_train_loader,
            reward_fn,
            actor_optim,
            critic_optim,
            max_grad_norm,
            epoch,
            run_io,
        )

        # Save models for checkpointing
        epoch_dir = os.path.join(run_io.checkpoint_dir, f"epoch-{epoch}")
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        torch.save(actor.state_dict(), os.path.join(epoch_dir, "actor.pt"))
        torch.save(critic.state_dict(), os.path.join(epoch_dir, "critic.pt"))

        # run validation
        mean_val_reward = validate(
            val_loader, actor, reward_fn, render_fn, num_plot=5, run_io=run_io
        )

        # Save best models
        if mean_val_reward < best_reward:
            best_reward = mean_val_reward
            torch.save(actor.state_dict(), run_io.actor_path)
            torch.save(critic.state_dict(), run_io.critic_path)

        # Logging model evaluation
        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        run_io.log(
            "Epoch %d, Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, "
            "took: %2.4fs (%2.4fs / 100 batches)\n\n"
            % (
                epoch,
                mean_loss,
                mean_reward,
                mean_val_reward,
                time.time() - epoch_start,
                np.mean(times),
            )
        )

        # increment epoch in curriculum
        curriculum.increment_epoch()


def main_tsp(args, run_io):
    """Main method for training/testing model for tsp task."""
    # creating curriculum specified by args.curriculum
    curriculum = get_indexed_curriculum(args.curriculum)(
        args.epochs,
        args.num_nodes,
        args.train_size,
        args.val_size,
        args.seed,
        debug=DEBUG,
    )

    # creating models on cpu
    actor = DRL4TSP(
        STATIC_SIZE,
        TSP_DYNAMIC_SIZE,
        args.hidden_size,
        None,  # update dynamic is None
        tsp.update_mask,
        args.num_layers,
        args.dropout,
    )

    critic = StateCritic(STATIC_SIZE, TSP_DYNAMIC_SIZE, args.hidden_size)

    # load models to cpu if necessary
    if args.load:
        saved_actor_state_dict = torch.load(
            run_io.actor_path, map_location=torch.device("cpu")
        )
        actor.load_state_dict(saved_actor_state_dict)

        saved_critic_state_dict = torch.load(
            run_io.critic_path, map_location=torch.device("cpu")
        )
        critic.load_state_dict(saved_critic_state_dict)

    # sending to proper device
    actor.to(DEVICE)
    critic.to(DEVICE)

    if args.mode == "train":
        # train only for train mode
        train_curriculum(
            actor,
            critic,
            curriculum,
            tsp.reward,
            tsp.render,
            args.epochs,
            args.batch_size,
            args.actor_lr,
            args.critic_lr,
            args.max_grad_norm,
            run_io,
        )
    elif args.mode == "test":
        # test only for test mode
        val_loader = DataLoader(
            curriculum.get_val_dataset(),
            args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_tsp(
            val_loader,
            actor,
            tsp.reward,
            tsp.render,
            num_plot=5,
            run_io=run_io,
        )
    elif args.mode == "all":
        train_curriculum(
            actor,
            critic,
            curriculum,
            tsp.reward,
            tsp.render,
            args.epochs,
            args.batch_size,
            args.actor_lr,
            args.critic_lr,
            args.max_grad_norm,
            run_io,
        )
        val_loader = DataLoader(
            curriculum.get_val_dataset(),
            args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_tsp(
            val_loader,
            actor,
            tsp.reward,
            tsp.render,
            num_plot=5,
            run_io=run_io,
        )


def main_vrp(args, run_io):
    # this function has not been refactored
    # this function has not been refactored
    # this function has not been refactored
    # this function has not been refactored
    # this function has not been refactored
    # this function has not been refactored
    # this function has not been refactored
    # this function has not been refactored
    # this function has not been refactored
    # this function has not been refactored

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    # Determines the maximum amount of load for a vehicle based on num nodes

    max_load = VRP_LOAD_DICT[args.num_nodes]

    train_data = VehicleRoutingDataset(
        args.train_size, args.num_nodes, max_load, VRP_MAX_DEMAND, args.seed
    )

    valid_data = VehicleRoutingDataset(
        args.valid_size,
        args.num_nodes,
        max_load,
        VRP_MAX_DEMAND,
        args.seed + 1,
    )

    actor = DRL4TSP(
        STATIC_SIZE,
        VRP_DYNAMIC_SIZE,
        args.hidden_size,
        train_data.update_dynamic,
        train_data.update_mask,
        args.num_layers,
        args.dropout,
    ).to(DEVICE)

    critic = StateCritic(STATIC_SIZE, VRP_DYNAMIC_SIZE, args.hidden_size).to(
        DEVICE
    )

    kwargs = vars(args)
    kwargs["train_data"] = train_data
    kwargs["valid_data"] = valid_data
    kwargs["reward_fn"] = vrp.reward
    kwargs["render_fn"] = vrp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, "actor.pt")
        actor.load_state_dict(torch.load(path, DEVICE))

        path = os.path.join(args.checkpoint, "critic.pt")
        critic.load_state_dict(torch.load(path, DEVICE))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = VehicleRoutingDataset(
        args.valid_size,
        args.num_nodes,
        max_load,
        VRP_MAX_DEMAND,
        args.seed + 2,
    )

    test_dir = "test"
    test_loader = DataLoader(
        test_data, args.batch_size, shuffle=False, num_workers=0
    )
    out = validate(
        test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5
    )

    print("Average tour length: ", out)


def main(args, run_io):
    """Main method for script."""
    if args.task == "tsp":
        main_tsp(args, run_io)
    elif args.task == "vrp":
        main_vrp(args, run_io)
    else:
        print("ERROR: Something has gone horribly wrong.")
        sys.exit(-1)


if __name__ == "__main__":
    # parsing args
    args = parse_arguments()

    # checking arguments for validity
    check_args_valid(args)

    # taking care of debug stuff
    if args.debug:
        args.train_size = 5
        args.val_size = 3
        args.batch_size = 2
        args.epochs = 20

        DEBUG = True
        ORTOOLS_TSP_TIMEOUT = 1
    else:
        # debug mode uses default backend, production mode does not
        matplotlib.use("Agg")

    # setting up current runIO
    curr_run_io = RunIO(args.task, args.num_nodes, args.run_name, args.load)

    if DEVICE == "cuda":
        with torch.cuda.device(args.device_id):
            main(args, curr_run_io)
    else:
        main(args, curr_run_io)
