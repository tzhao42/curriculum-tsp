#!/bin/bash

# Debugging script for debugging on my machine

cd ../src
# python3 main.py --mode all --curriculum 4 --run-name debug-0 --debug --regen
python3 main.py --mode test --curriculum 4 --debug --run-name debug --load tsp-20-1.00-0.00-0.00-20201114T193737

