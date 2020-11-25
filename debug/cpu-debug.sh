#!/bin/bash

# Debugging script for debugging on my machine

cd ../src
python3 main.py --mode train --run-name debug-0 --load tsp-20-0.00-0.00-1.00-20201114T193737 --proportions 1.00 0.00 0.00 --debug
# python3 main.py --mode test --run-name debug-0 --load tsp-20-0.00-0.00-1.00-20201114T193737 --proportions 1.00 0.00 0.00 --debug

# python3 main.py --debug --mode train --load tsp-20-1.00-0.00-0.00-20201114T193737 --run-name full-model-eval --proportions 1.00 0.00 0.00 &
# python3 main.py --debug --mode train --load tsp-20-0.85-0.15-0.00-20201114T193737 --run-name full-model-eval --proportions 0.85 0.15 0.00 &
# python3 main.py --debug --mode train --load tsp-20-0.50-0.50-0.00-20201114T193737 --run-name full-model-eval --proportions 0.50 0.50 0.00 &
# python3 main.py --debug --mode train --load tsp-20-0.15-0.85-0.00-20201114T193737 --run-name full-model-eval --proportions 0.15 0.85 0.00 &
# python3 main.py --debug --mode train --load tsp-20-0.00-1.00-0.00-20201114T193737 --run-name full-model-eval --proportions 0.00 1.00 0.00 &
# python3 main.py --debug --mode train --load tsp-20-0.85-0.00-0.15-20201114T193737 --run-name full-model-eval --proportions 0.85 0.00 0.15 &
# python3 main.py --debug --mode train --load tsp-20-0.50-0.00-0.50-20201114T193737 --run-name full-model-eval --proportions 0.50 0.00 0.50 &
# python3 main.py --debug --mode train --load tsp-20-0.15-0.00-0.85-20201114T193737 --run-name full-model-eval --proportions 0.15 0.00 0.85 &
# python3 main.py --debug --mode train --load tsp-20-0.00-0.00-1.00-20201114T193737 --run-name full-model-eval --proportions 0.00 0.00 1.00



