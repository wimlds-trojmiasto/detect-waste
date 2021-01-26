python3 efficientdet/train.py "/dih4/dih4_2/wimlds/data/" \
        --model tf_efficientdet_d2 --batch-size 4 --decay-rate 0.95 \
        --lr .001 --workers 4 --warmup-epochs 5 --model-ema --dataset binary \
        --pretrained --num-classes 1 --color-jitter 0.1 --reprob 0.2 --epochs 40 \
