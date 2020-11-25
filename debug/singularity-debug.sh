#!/bin/bash

SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/ortools-pytorch-gpu.simg"

cd ..
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --run-name debug-0 --device-id 0 --load tsp-20-0.00-0.00-1.00-20201114T193737 --proportions 0.00 0.00 1.00 --debug
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --run-name debug-1 --device-id 1 --load tsp-20-1.00-0.00-0.00-20201114T193737 --proportions 1.00 0.00 0.00 --debug

