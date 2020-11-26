"""Various curriculums."""

import torch

from constants import (
    LOG_DIR,
    DEVICE,
    DEBUG,
    NUM_TILES,
    ORTOOLS_TSP_TIMEOUT,
    STATIC_SIZE,
    TSP_DYNAMIC_SIZE,
    VRP_LOAD_DICT,
    VRP_MAX_DEMAND,
    VRP_DYNAMIC_SIZE,
)

from tasks import tsp, node_distrib
from tasks.node_distrib import (
    get_uniform_param
)
from tasks.tsp import TSPCurriculum


def get_uniform_curriculum(epochs, num_nodes, num_samples, seed):
    """Create the uniform curriculum."""
    curriculum = TSPCurriculum(num_nodes, num_samples, seed)

    unif_param = get_uniform_param(NUM_TILES)
    curriculum.add_stage(NUM_TILES, unif_param, epochs)

    curriculum.start()
    return curriculum

