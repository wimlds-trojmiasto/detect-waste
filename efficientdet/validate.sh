python3 efficientdet/validate.py "/dih4/dih4_2/wimlds/TACO-master/data/" \
        --model tf_efficientdet_d3 --batch-size 2 --num-classes 7 \
        --workers 104 --dataset TACOCfg \
        --checkpoint 'output/train/20201208-155432-tf_efficientdet_d3/model_best.pth.tar'
