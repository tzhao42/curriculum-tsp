#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:tesla-k80:2
#SBATCH -p cbmm

# command
module load openmind/singularity
hostname
nvidia-smi
singularity exec --nv /om2/user/tzhao/singularity/pytorch-gpu.simg 

python3 trainer.py --device_id 0 --run-name 1.00-0.00-0.00 --proportions 1.00 0.00 0.00

python3 trainer.py --device_id 0 --run-name 0.85-0.15-0.00 --proportions 0.85 0.15 0.00
python3 trainer.py --device_id 0 --run-name 0.50-0.50-0.00 --proportions 0.50 0.50 0.00
python3 trainer.py --device_id 0 --run-name 0.15-0.85-0.00 --proportions 0.15 0.85 0.00
python3 trainer.py --device_id 0 --run-name 0.00-1.00-0.00 --proportions 0.00 1.00 0.00

python3 trainer.py --device_id 1 --run-name 0.85-0.00-0.15 --proportions 0.85 0.00 0.15
python3 trainer.py --device_id 1 --run-name 0.50-0.00-0.50 --proportions 0.50 0.00 0.50
python3 trainer.py --device_id 1 --run-name 0.15-0.00-0.85 --proportions 0.15 0.00 0.85
python3 trainer.py --device_id 1 --run-name 0.00-0.00-1.00 --proportions 0.00 0.00 1.00



