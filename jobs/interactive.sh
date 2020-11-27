#!/bin/bash

srun -p normal -t 02:00:00 --mem=10G -N 1 -n 1 --gres=gpu:1 --constraint=12GB --pty bash
