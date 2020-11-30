#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=6GB
#SBATCH -p normal

# This script was generate automatically

SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/ortools-pytorch-gpu.simg"
STARTING_DIRECTORY="/om2/user/tzhao/6883/curriculum-tsp"

cd ${STARTING_DIRECTORY}
module load openmind/singularity
hostname
nvidia-smi

singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --run-name test-20-static-exp-15 --num-nodes 20 --epochs 20 --curriculum 15 --val-set tsp-20-val-1000.npy --load tsp-20-static-exp-15-20201129T003049 &
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --run-name test-20-static-exp-16 --num-nodes 20 --epochs 20 --curriculum 16 --val-set tsp-20-val-1000.npy --load tsp-20-static-exp-16-20201129T003049 &


sleep 2h
nvidia-smi
sleep 72h


