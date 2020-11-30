#!/bin/bash

# Debugging script for debugging on my machine

cd ../src
# python3 main.py --debug --num-nodes 100 --mode all --curriculum 0 --run-name debug-0 --regen
# python3 main.py --debug --num-nodes 100 --mode train --curriculum 2 --run-name debug-1
# python3 main.py --debug --num-nodes 100 --mode train --curriculum 2 --run-name debug-1 
# python3 main.py --debug --num-nodes 20 --mode test --curriculum 0 --run-name debug-2 --load tsp-20-loader-00000000T000000
# python3 main.py --debug --num-nodes 100 --mode test --run-name debug --val-set tsp-100-val-10000.npy --curriculum 0
# python3 main.py --debug --num-nodes 20 --mode train --run-name evaluation --curriculum 0

# python3 main.py --num-nodes 20 --mode test --run-name evaluation --val-set tsp-100-val-1000.npy --load tsp-20-evaluation-20201129T172637 --curriculum 0

python3 main.py --mode test --run-name test-20-static-exp-01 --num-nodes 20 --epochs 20 --curriculum 1 --val-set tsp-20-val-1000.npy --load tsp-20-evaluation-20201129T172637

