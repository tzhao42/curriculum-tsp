"""
Constants for operation.
"""

import os
import pathlib

# Directory containing 6883-vrp repo
# BASE_DIR = pathlib.Path(__file__).parent.absolute()
# Does this solve the bug?
BASE_DIR = pathlib.Path.cwd()

# Directory for logs
LOG_DIR = os.path.join(BASE_DIR, "log")

