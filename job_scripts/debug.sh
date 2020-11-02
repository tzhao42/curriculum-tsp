#!/bin/bash
# to be run on compute nodes only
# maybe need to bind stuff? not sure

singularity exec --nv /om2/user/tzhao/singularity/pytorch-gpu.simg python3 trainer.py --debug --task tsp --train-size 31 --valid-size 17 --run-name debug --device_id 0 --proportions 1 0 0 &

singularity exec --nv /om2/user/tzhao/singularity/pytorch-gpu.simg python3 trainer.py --debug --task tsp --train-size 31 --valid-size 17 --run-name debug --device_id 0 --proportions 1 0 0 &
