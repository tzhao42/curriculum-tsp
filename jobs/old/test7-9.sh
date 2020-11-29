#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
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

# singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode all --run-name static-exp-7 --curriculum 7 &
# singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode all --run-name static-exp-8 --curriculum 8 &
# singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode all --run-name static-exp-9 --curriculum 9 &

singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-static-exp-7-20201127T012939 --run-name test-static-exp-7 --curriculum 7 &
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-static-exp-8-20201127T012939 --run-name test-static-exp-8 --curriculum 8 &
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-static-exp-9-20201127T012939 --run-name test-static-exp-9 --curriculum 9 &

sleep 3h
nvidia-smi
sleep 48h

