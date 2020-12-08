python3 efficientdet/train.py "/dih4/dih4_2/wimlds/TACO-master/data/" \
        --model tf_efficientdet_d0 --batch-size 8 \
        --lr .09 --workers 104 --warmup-epochs 5 --model-ema --dataset TACOCfg \
        --pretrained --num-classes 7 --color-jitter 0.3 --reprob 0.2 \
