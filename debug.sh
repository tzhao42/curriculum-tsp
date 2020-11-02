#!/bin/bash

python3 trainer.py --debug --task tsp --train-size 31 --valid-size 17 --run-name debug --device_id 0 --proportions 1 0 0

python3 trainer.py --debug --task tsp --train-size 31 --valid-size 17 --run-name debug --device_id 1 --proportions 1 0 0
