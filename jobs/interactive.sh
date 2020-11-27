#!/bin/bash

srun -p cbmm -t 01:00:00 --mem=10G -N 1 -n 1 --gres=gpu:tesla-k80:2 --pty bash
