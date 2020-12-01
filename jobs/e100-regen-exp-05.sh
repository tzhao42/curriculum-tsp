#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=12GB
#SBATCH -p cbmm

# This script was generate automatically

SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/ortools-pytorch-gpu.simg"
STARTING_DIRECTORY="/om2/user/tzhao/6883/curriculum-tsp"

cd ${STARTING_DIRECTORY}
module load openmind/singularity
hostname
nvidia-smi

singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode all --run-name regen-exp-05-epochs-30 --num-nodes 100 --epochs 30 --curriculum 5 --regen --val-set tsp-100-val-1000.npy &


sleep 2h
nvidia-smi
sleep 72h


