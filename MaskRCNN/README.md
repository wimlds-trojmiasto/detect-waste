# PyTorch-MaskRCNN

A PyTorch implementation of simple Mask R-CNN.

This repository is a toy example of Mask R-CNN with two features:
- It is pure python code and can be run immediately using PyTorch 1.5.1 and trochvision 0.8.2,
- Simplified construction and easy to understand how the model works based on [tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

The code is based largely on [TorchVision](https://github.com/pytorch/vision).

## Requirements

- **Windows** or **Linux**, with **Python ≥ 3.7**

- **[PyTorch](https://pytorch.org/) ≥ 1.5.0**

- **matplotlib** - visualizing images and results

- **[pycocotools](https://github.com/cocodataset/cocoapi)** - for COCO dataset and evaluation; Windows version is [here](https://github.com/philferriere/cocoapi)

## Datasets

This repository supports detect-waste datasets (usualy COCO based annotations) with one class - waste.

## Training

Simply run:

```
python train.py --num_epochs 20 --gpu_id 2 --output_dir /dih4/dih4_2/wimlds/<user_name>
```
or modify the parameters in ```run.sh```, and run:

```
bash ./run.sh
```

Note: This is a simple model and only supports one gpu (not distribiuted training).

## Evaluation

- Modify the parameters in ```eval.ipynb``` to test the model: TBA

## Performance

The model utilizes part of TorchVision's weights, which is pretrained on COCO dataset.
Test on Multi detect-waste Segmentation val, on 1 RTX 2080Ti GPU:

|     model  | backbone  | epoch | bbox AP@0.5 | bbox AP@0.5:0.95 | mask AP@0.5 | mask AP@0.5:0.95 |
| ---------- | --------- | ----- | ----------- | ---------------- | ----------- | ---------------- |
| Mask R-CNN | ResNet 50 |    2  |  0.25       |       0.14       |    0.21     |      0.12        |
