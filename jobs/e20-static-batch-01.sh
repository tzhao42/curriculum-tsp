#!/bin/bash
#SBATCH -t 72:00:00
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

singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode all --run-name static-exp-02-epochs-30 --num-nodes 20 --epochs 30 --curriculum 2 --val-set tsp-20-val-1000.npy &
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode all --run-name static-exp-03-epochs-30 --num-nodes 20 --epochs 30 --curriculum 3 --val-set tsp-20-val-1000.npy &


sleep 2h
nvidia-smi
sleep 72h


