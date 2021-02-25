#!/bin/bash
epochs=26
gpu=2
path='/dih4/dih4_2/wimlds/smajchrowska'
img='/dih4/dih4_2/wimlds/data/all_detect_images'
anno='../annotations/annotations_binary_mask0_all'

python train.py --num_epochs ${epochs} --gpu_id ${gpu} --output_dir ${path} \
                --anno_name ${anno} --images_dir ${img} --neptune
