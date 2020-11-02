#!/bin/bash

SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/ortools-pytorch-gpu.simg"

singularity exec --nv ${SINGULARITY_CONTAINER} python3 trainer.py --debug --device_id 0 --run-name debug0 --train-size 31 --valid-size 19 --proportions 1.00 0.00 0.00 &
singularity exec --nv ${SINGULARITY_CONTAINER} python3 trainer.py --debug --device_id 1 --run-name debug1 --train-size 31 --valid-size 19 --proportions 0.85 0.00 0.15 &

