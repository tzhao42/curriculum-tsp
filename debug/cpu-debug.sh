#!/bin/bash

# Debugging script for debugging on my machine

cd ../src
python3 main.py --debug --mode all --curriculum 0 --run-name debug-0 --regen
python3 main.py --debug --mode test --curriculum 4 --run-name debug-1 --load tsp-20-loader-00000000T000000

