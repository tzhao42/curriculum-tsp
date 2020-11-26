#!/bin/bash

# Debugging script for debugging on my machine

cd ../src
python3 main.py --mode train --run-name debug-0 --debug
python3 main.py --mode test --run-name debug-0 --debug
