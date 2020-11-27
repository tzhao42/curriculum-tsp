#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 4
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

# singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode all --run-name static-exp-4 --curriculum 4 &
# singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode all --run-name static-exp-5 --curriculum 5 &
# singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode all --run-name static-exp-6 --curriculum 6 &

singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-static-exp-4-20201127T012931 --run-name test-static-exp-4 --curriculum 4 &
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-static-exp-5-20201127T012930 --run-name test-static-exp-5 --curriculum 5 &
singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --mode test --load tsp-20-static-exp-6-20201127T012931 --run-name test-static-exp-6 --curriculum 6 &

sleep 3h
nvidia-smi
sleep 48h

