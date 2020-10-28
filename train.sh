#!/bin/bash

nohup python3 trainer.py --run-name 1.00-0.00-0.00 --proportions 1.00 0.00 0.00 &

nohup python3 trainer.py --run-name 0.85-0.15-0.00 --proportions 0.85 0.15 0.00 &
nohup python3 trainer.py --run-name 0.50-0.50-0.00 --proportions 0.50 0.50 0.00 &
nohup python3 trainer.py --run-name 0.15-0.85-0.00 --proportions 0.15 0.85 0.00 &
nohup python3 trainer.py --run-name 0.00-1.00-0.00 --proportions 0.00 1.00 0.00 &

nohup python3 trainer.py --run-name 0.85-0.00-0.15 --proportions 0.85 0.00 0.15 &
nohup python3 trainer.py --run-name 0.50-0.00-0.50 --proportions 0.50 0.00 0.50 &
nohup python3 trainer.py --run-name 0.15-0.00-0.85 --proportions 0.15 0.00 0.85 &
nohup python3 trainer.py --run-name 0.00-0.00-1.00 --proportions 0.00 0.00 1.00 &

