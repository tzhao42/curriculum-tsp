#!/bin/bash

# srun -p cbmm -t 02:00:00 --mem=10G -N 1 -n 1 --gres=gpu:tesla-k80:1 --pty bash
srun -p normal -t 02:00:00 --mem=10G -N 1 -n 1 --gres=gpu:tesla-k80:1 --pty bash
