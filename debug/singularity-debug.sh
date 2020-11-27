#!/bin/bash

SINGULARITY_CONTAINER="/om2/user/tzhao/singularity/ortools-pytorch-gpu.simg"

cd ..
# singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --debug --mode all --run-name debug-0 --curriculum 7
# singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --debug --mode all --run-name debug-1 --curriculum 0 --regen 

singularity exec --nv ${SINGULARITY_CONTAINER} python3 src/main.py --num-nodes 100 --mode all --run-name debug-resource-usage --curriculum 0 --regen &

