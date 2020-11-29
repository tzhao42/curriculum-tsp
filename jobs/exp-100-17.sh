#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=12GB
#SBATCH -p normal

# Setting up singularity variable
SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/ortools-pytorch-gpu.simg"
STARTING_DIRECTORY="/om2/user/tzhao/6883/6883-vrp"

cd ${STARTING_DIRECTORY}
module load openmind/singularity
hostname
nvidia-smi

singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --num-nodes 100 --mode all --run-name static-exp-17 --curriculum 17 &

sleep 3h
nvidia-smi
sleep 72h