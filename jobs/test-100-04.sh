#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=12GB
#SBATCH -p normal

# This script was generate automatically

SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/ortools-pytorch-gpu.simg"
STARTING_DIRECTORY="/om2/user/tzhao/6883/curriculum-tsp"

cd ${STARTING_DIRECTORY}
module load openmind/singularity
hostname
nvidia-smi

singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --run-name test-100-static-exp-04 --num-nodes 100 --epochs 20 --curriculum 4 --val-set tsp-100-val-1000.npy --load tsp-100-static-exp-4-20201129T003051 &


sleep 2h
nvidia-smi
sleep 72h


