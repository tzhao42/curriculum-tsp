#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --constraint=12GB
#SBATCH -p normal

# Setting up singularity variable
SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/ortools-pytorch-gpu.simg"
STARTING_DIRECTORY="/om2/user/tzhao/6883/curriculum-tsp"

cd ${STARTING_DIRECTORY}
module load openmind/singularity
hostname
nvidia-smi
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-1.00-0.00-0.00-20201114T193737 --run-name full-model-eval --proportions 1.00 0.00 0.00 &
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-0.85-0.15-0.00-20201114T193737 --run-name full-model-eval --proportions 0.85 0.15 0.00 &
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-0.50-0.50-0.00-20201114T193737 --run-name full-model-eval --proportions 0.50 0.50 0.00 &

sleep 24h

