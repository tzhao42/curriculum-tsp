"""
Constants for operation.
"""

import os
import pathlib

import torch

# Directory containing 6883-vrp repo
BASE_DIR = pathlib.Path(__file__).parent.absolute()

# Directory for logs
LOG_DIR = os.path.join(BASE_DIR, "log")

# Devices available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
