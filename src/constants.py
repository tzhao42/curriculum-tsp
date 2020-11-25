"""
Constants for operation.
"""

import os
import pathlib

import torch

# Directory containing 6883-vrp repo
BASE_DIR = pathlib.Path(__file__).parent.absolute().parents[0]

# Directory for logs
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Device available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameterization for graph
# Parmaeterize graph as an N_TILES x N_TILES grid of tiles, each of which has
# a weighted probability of having points spawn within it.
N_TILES = 8 

# Parameters for models
STATIC_SIZE = 2  # (x, y)

# TSP
TSP_DYNAMIC_SIZE = 1  # dummy for compatibility

# VRP
VRP_LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
VRP_MAX_DEMAND = 9
VRP_DYNAMIC_SIZE = 2  # (load, demand)

# Debug flag
DEBUG = False
