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

from constants import BASE_DIR, LOG_DIR
from model import DRL4TSP, Encoder
from tasks import tsp, vrp, tsp_or_tools
from tasks.tsp import TSPDataset
from tasks.vrp import VehicleRoutingDataset

from tasks.tsp_or_tools import get_batched_or_tsp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output


class Logger:
    """
    Logging object for training runs.
    """

    def __init__(self, task, nodes, run_name):
        """
        Initializes a logger object.
        Args:
            task: string, "tsp" or "vrp"
            nodes: int, one of: [10, 20, 50, 100]
            run_name: string, describes current run
        """
        assert task in {"tsp", "vrp"}
        assert nodes in {10, 20, 50, 100}
        assert isinstance(run_name, str)

        # initializing instance attributes
        self._task = task
        self._nodes = nodes
        self._run_name = run_name

        # unpacking current datetime
        curr_datetime = datetime.datetime.now()
        curr_year = curr_datetime.year
        curr_month = "{:02}".format(curr_datetime.month)
        curr_day = "{:02}".format(curr_datetime.day)
        curr_hour = "{:02}".format(curr_datetime.hour)
        curr_minute = "{:02}".format(curr_datetime.minute)
        curr_second = "{:02}".format(curr_datetime.second)
        self._time = (
            f"{curr_year}{curr_month}{curr_day}T{curr_hour}{curr_minute}{curr_second}"
        )

        # superdirectory name
        self._superdir_name = f"{task}-{nodes}"

        # namestring
        self._name = f"{self._task}-{self._nodes}-{self._run_name}-{self._time}"

        # creating superdirectory
        self._superdir = os.path.join(LOG_DIR, self._superdir_name)
        if not os.path.exists(self._superdir):
            os.makedirs(self._superdir)

        # creating directory
        self.dir = os.path.join(self._superdir, self._name)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # creating log file
        self._logfile = os.path.join(self.dir, "log.log")
        with open(self._logfile, "a+") as f:
            f.write(f"Current device: {device}\n")

        # creating validate and checkpoint subdirs
        self.validate_dir = os.path.join(self.dir, "validate")
        if not os.path.exists(self.validate_dir):
            os.makedirs(self.validate_dir)

        self.checkpoint_dir = os.path.join(self.dir, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def log(self, message):
        """Writes a line to the log file."""
        with open(self._logfile, "a+") as f:
            f.write(message)


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
                    % (epoch, batch_idx, len(train_loader), mean_reward, mean_loss, times[-1])
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

    # Goals from paper:
    # TSP20, 3.97
    # TSP50, 6.08
    # TSP100, 8.44

    # checking if debug
    if args.debug:
        args.train_size = 10
        args.valid_size = 10

    # initializing logger
    logger = Logger(args.task, args.num_nodes, args.run_name)

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

    parser = argparse.ArgumentParser(description="Combinatorial Optimization")

    # their arguments
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--task", default="tsp")
    parser.add_argument("--nodes", dest="num_nodes", default=20, type=int)
    parser.add_argument("--actor_lr", default=5e-4, type=float)
    parser.add_argument("--critic_lr", default=5e-4, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden", dest="hidden_size", default=128, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--layers", dest="num_layers", default=1, type=int)
    parser.add_argument("--train-size", default=1000000, type=int)
    parser.add_argument("--valid-size", default=1000, type=int)

    # our arguments
    parser.add_argument("--run-name", default="tsp", type=str)
    parser.add_argument("--proportions", nargs=3, default=None, type=float)
    parser.add_argument("--device_id", default=0, type=int)

    # debug flag: short circuits training cycle
    parser.add_argument(
        "--debug", dest="debug", default=False, action="store_true")

    args = parser.parse_args()

    # print('NOTE: SETTTING CHECKPOINT: ')
    # args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    # print(args.checkpoint)

    with torch.cuda.device(args.device_id):
        if args.task == "tsp":
            train_tsp(args)
        elif args.task == "vrp":
            train_vrp(args)
        else:
            raise ValueError("Task <%s> not understood" % args.task)
