#!/bin/bash

# Debugging script for debugging on my machine

cd ../src

python3 main.py --debug --mode all --run-name regen-exp-02-epochs-30 --num-nodes 20 --epochs 30 --curriculum 2 --regen --val-set tsp-20-val-1000.npy

