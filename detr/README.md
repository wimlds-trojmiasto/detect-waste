# **DE⫶TR** (PyTorch) for waste detection
PyTorch training code and pretrained models for **DETR** (**DE**tection **TR**ansformer).
Authors replace the full complex hand-crafted object detection pipeline with a Transformer, and match Faster R-CNN with a ResNet-50, obtaining **42 AP** on COCO using half the computation power (FLOPs) and the same number of parameters. Inference in 50 lines of PyTorch.

**What it is**. Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. 
Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

For details see [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko.

# Implementation
A PyTorch **DE⫶TR** for detecting waste using implementation from  [Official facebookresearch implementation](https://github.com/facebookresearch/detr).

Authors provided baseline DETR and DETR-DC5 models, and plan to include more in future.
AP is computed on COCO 2017 val5k, and inference time is over the first 100 val5k COCO images,
with torchscript transformer.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>schedule</th>
      <th>inf_time</th>
      <th>box AP</th>
      <th>url</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>500</td>
      <td>0.036</td>
      <td>42.0</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>500</td>
      <td>0.083</td>
      <td>43.3</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50-dc5_log.txt">logs</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>500</td>
      <td>0.050</td>
      <td>43.5</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DETR-DC5</td>
      <td>R101</td>
      <td>500</td>
      <td>0.097</td>
      <td>44.9</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101-dc5_log.txt">logs</a></td>
      <td>232Mb</td>
    </tr>
  </tbody>
</table>

COCO val5k evaluation results can be found in this [gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918).

The models are also available via torch hub,
to load DETR R50 with pretrained weights simply do:
```python
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
```

# Dataset & model modifications
* we use TACO dataset with additional annotated data from detect-waste.

We expect the directory structure to be the following:
```
path/to/dato/
  annotations/  # annotation json files
  train/        # train images
  val/          # val images
```
You can modify `datasets/coco.py` build function to add new dataset and another format of paths for coco annotations type.

## Model details
* Optimizer: AdamW or LaProp
* Number of class: 7 (paper, metals and plastics, bio, other, non-recycle, glass, unknown) or 1 (litter)
* Backbone: ResNet50
* Num queries: 100 (like in official Detr it coressponds to max number of instances per images - this should not be changed if we finetuned)
* Eos coef: 0.1 (like in official Detr mean number of instances per image - this should not be changed if we finetuned)
* 300 epochs at lr 1e-4 with lr_drop to 1e-5 at 200

# Usage - Object detection
There are no extra compiled components in DETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally, and then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

(optional) to work with panoptic install panopticapi:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Training
To train baseline DETR on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/taco --dataset_file taco --dataset_mode one --output_dir wimlds_1 --resume detr-r50-e632da11.pth
```
... with one gpu for mixed dataset of waste:

```
python3 main.py --coco_path /path/to/taco --dataset_file taco --dataset_mode one --output_dir multi_1 --resume detr-r50-e632da11.pth
```
or `--dataset_mode wimlds` for 8 classes example.

We train DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales and crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.

## Evaluation
To evaluate DETR R50 with a single GPU run:
```
python main.py --batch_size 2 --no_aux_loss --eval --resume wimlds_1/checkpoint0099.pth --coco_path /dih4/dih4_2/wimlds/data/all_detect_images/ --dataset_file multi
```

## Results on TACO
Our resuts are presented in ```./notebooks``` directory.

| model | backbone  | Dataset       | # classes| bbox AP@0.5 | bbox AP@0.5:0.95 | mask AP@0.5 | mask AP@0.5:0.95 |
| :---: | :-------: | :-----------: | :-------:| :---------: | :--------------: | :---------: | :--------------: |
| DETR    | ResNet 50 |   TACO bboxes | 1        |    46.50    |       24.34      |      x      |  x               |
| DETR    | ResNet 50 |   TACO bboxes | 7        |    6.69     |       3.23       |      x      |  x               |
| DETR    | ResNet 50 |   *Multi      | 1        |    50.68    |       27.69      |      **54.80      |  **32.17               |
