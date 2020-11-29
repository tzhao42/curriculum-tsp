#!/bin/bash

# Debugging script for debugging on my machine

cd ../src
# python3 main.py --debug --num-nodes 100 --mode all --curriculum 0 --run-name debug-0 --regen
# python3 main.py --debug --num-nodes 100 --mode train --curriculum 2 --run-name debug-1
# python3 main.py --debug --num-nodes 100 --mode train --curriculum 2 --run-name debug-1 
# python3 main.py --debug --num-nodes 20 --mode test --curriculum 0 --run-name debug-2 --load tsp-20-loader-00000000T000000
python3 main.py --debug --num-nodes 20 --mode all --run-name static-exp-18 --curriculum 18 --epochs 30

