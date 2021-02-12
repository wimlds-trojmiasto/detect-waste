#!/bin/bash
shift
python -m torch.distributed.launch --nproc_per_node=4 efficientdet/train.py "/dih4/dih4_2/wimlds/TACO-master/data/" \
        --model tf_efficientdet_d3 --batch-size 2 --decay-rate 0.9 \
        --lr .001 --workers 4 --warmup-epochs 5 --model-ema --dataset TACOCfg \
        --pretrained --num-classes 7 --color-jitter 0.3 --reprob 0.2 \

