python3 efficientdet/train.py "/dih4/dih4_2/wimlds/TACO-master/data/" \
        --model tf_efficientdet_d2 --batch-size 3 --decay-rate 0.9 \
        --lr .001 --workers 104 --warmup-epochs 5 --model-ema --dataset TACOCfg \
        --pretrained --num-classes 7 --color-jitter 0.1 --reprob 0.2 \
