#!/bin/bash
iters=20
gpu=2
path='/dih4/dih4_2/wimlds/smajchrowska'

python train.py --num_epochs ${epochs} --gpu_id ${gpu} --output_dir ${path} --neptune
