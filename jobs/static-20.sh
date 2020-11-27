#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=6G
#SBATCH --gres=gpu:1
#SBATCH --constraint=4GB
#SBATCH -p normal

# Setting up singularity variable
SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/ortools-pytorch-gpu.simg"
STARTING_DIRECTORY="/om2/user/tzhao/6883/6883-vrp"

cd ${STARTING_DIRECTORY}
module load openmind/singularity
hostname
nvidia-smi

singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode all --run-name static-uniform  --curriculum 0 & 

sleep 2h
nvidia-smi
sleep 48h

