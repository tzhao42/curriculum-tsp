#!/bin/bash

srun -p normal -t 04:00:00 --mem=12G -N 1 -n 2 --gres=gpu:1 --constraint=12GB --pty bash
