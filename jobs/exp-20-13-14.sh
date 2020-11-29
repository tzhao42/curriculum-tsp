#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --constraint=6GB
#SBATCH -p normal

# Setting up singularity variable
SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/ortools-pytorch-gpu.simg"
STARTING_DIRECTORY="/om2/user/tzhao/6883/6883-vrp"

cd ${STARTING_DIRECTORY}
module load openmind/singularity
hostname
nvidia-smi

singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --num-nodes 20 --mode all --run-name static-exp-13 --curriculum 13 &
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --num-nodes 20 --mode all --run-name static-exp-14 --curriculum 14 &

sleep 3h
nvidia-smi
sleep 48h

