#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH -p normal

# Setting up singularity variable
SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/ortools-pytorch-gpu.simg"
STARTING_DIRECTORY="/om2/user/tzhao/6883/6883-vrp"

cd ${STARTING_DIRECTORY}
module load openmind/singularity
hostname
nvidia-smi
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-0.50-0.00-0.50-20201114T193737 --run-name full-model-eval --proportions 0.50 0.00 0.50
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-0.15-0.00-0.85-20201114T193737 --run-name full-model-eval --proportions 0.15 0.00 0.85
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-0.00-0.00-1.00-20201114T193737 --run-name full-model-eval --proportions 0.00 0.00 1.00

sleep 24h

