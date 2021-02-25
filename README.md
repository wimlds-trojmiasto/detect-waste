# Detect waste
AI4Good project for detecting waste in environment
[www.detectwaste.ml](www.detectwaste.ml)

## Data download (WIP)
* TACO bboxes - in progress. TACO dataset can be downloaded (http://tacodataset.org/)[here]. TACO bboxes will be avaiable for download soon.

    Clone Taco repository
        `git clone https://github.com/pedropro/TACO.git`

    Install requirements
        `pip3 install -r requirements.txt`

    Download annotated data
        `python3 download.py`

* [UAVVaste](https://github.com/UAVVaste/UAVVaste)

    Clone UAVVaste repository
        `git clone https://github.com/UAVVaste/UAVVaste.git`
    
    Install requirements
        `pip3 install -r requirements.txt`
    
    Download annotated data
        `python3 main.py`

* [TrashCan 1.0](https://conservancy.umn.edu/handle/11299/214865)

    Download directly from web
    `wget https://conservancy.umn.edu/bitstream/handle/11299/214865/dataset.zip?sequence=12&isAllowed=y`

* [TrashICRA](https://conservancy.umn.edu/handle/11299/214366)

    Download directly from web
    `wget https://conservancy.umn.edu/bitstream/handle/11299/214366/trash_ICRA19.zip?sequence=12&isAllowed=y`

* [MJU-Waste](https://github.com/realwecan/mju-waste/) 

    Download directly from [google drive](https://drive.google.com/file/d/1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny/view)

* [Drinking Waste Classification](https://www.kaggle.com/arkadiyhacks/drinking-waste-classification)

    In order to download you must first authenticate using a kaggle API token. Read about it [here](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication)

    `kaggle datasets download -d arkadiyhacks/drinking-waste-classification`

* [Wade-ai](https://github.com/letsdoitworld/wade-ai/tree/master/Trash_Detection)

    Clone wade-ai repository
        `git clone https://github.com/letsdoitworld/wade-ai.git`
    
    For coco annotation check: [majsylw/wade-ai/tree/coco-annotation] (https://github.com/majsylw/wade-ai/tree/coco-annotation/Trash_Detection/trash/dataset)

For more datasets check: [waste-datasets-review](https://github.com/AgaMiko/waste-datasets-review)

## Data preprocessing

### Multiclass training
To train only on TACO dataset with detect-waste classes:
* run *annotations_preprocessing.py*

    `python3 annotations_preprocessing.py`

    new annotations will be saved in *annotations/annotations_train.json* and *annotations/annotations_test.json*

### Single class training

To train on one or multiple datasets on a single class:

* run *annotations_preprocessing_multi.py*

    `python3 annotations_preprocessing_multi.py`

    new annotations will be split and saved in *annotations/binary_mixed_train.json* and *annotations/binary_mixed_test.json*

    Example bash file is in **annotations_preprocessing_multi.sh** and can be run by

    `bash annotations_preprocessing_multi.sh`

Script will automaticlly split all datasets to train and test set with MultilabelStratifiedShuffleSplit. Then it will convert datasets to one class - litter. Finally all datasets will be concatenated to form single train and test files *annotations/binary_mixed_train.json* and *annotations/binary_mixed_test.

# Models

To read more about past waste detection works check [litter-detection-review](https://github.com/majsylw/litter-detection-review)

* ### EfficientDet (WIP)

    To train EfficientDet check `efficientdet/README.md`
    
    For implementation details see [efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch) by Ross Wightman.

* ### DETR

    To train detr check `detr/README.md` (WIP)

    PyTorch training code and pretrained models for **DETR** (**DE**tection **TR**ansformer).
    Authors replaced the full complex hand-crafted object detection pipeline with a Transformer, and matched Faster R-CNN with a ResNet-50, obtaining **42 AP** on COCO using half the computation power (FLOPs) and the same number of parameters. Inference in 50 lines of PyTorch.

    For implementation details see [End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr) by Facebook.

* ### Mask R-CNN
    To train Mask R-CNN check `MaskRCNN/README.md`

    Our implementation based on [tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

* ### Fast R-CNN
    To train Fast R-CNN check `FastRCNN/README.md`

* ### Classification with ResNet50 and EfficientNet
   To train choosen model check `classifier/README.md`

## Our results

### Detection/Segmentation task
| model  | backbone  | Dataset       | # classes | bbox AP@0.5 | bbox AP@0.5:0.95 | mask AP@0.5 | mask AP@0.5:0.95 |
| :-----:| :-------: | :-----------: | :-------: | :---------: | :--------------: | :---------: | :--------------: |
| DETR    | ResNet 50 |   TACO bboxes | 1        |    46.50    |       24.34      |      x      |  x               |
| DETR    | ResNet 50 |   TACO bboxes | 7        |    6.69     |       3.23       |      x      |  x               |
| DETR    | ResNet 50 |   *Multi       | 1        |    50.68    |       27.69      |      **54.80      |  **32.17               |
| Mask R-CNN  | ResNet 50    |  *Multi   |  1    |    27.95 |       16.49   |    23.05     |    12.94       |
| Mask R-CNN  | ResNetXt 101 |  *Multi   |  1    |    19.70 |       6.20    |    24.70     |    13.20       |
| EfficientDet-D2 | EfficientNet-B2 |    Taco bboxes  |  1    |    61.05  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    Taco bboxes  |  7    |    18.78  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    Drink-waste  |  4    |    99.60  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    MJU-Waste    |  1    |    97.74  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    TrashCan v1  |  8    |    91.28  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    Wade-AI      |  1    |    33.03  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    UAVVaste     |  1    |    79.90  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    Trash ICRA19 |  7    |    9.47   |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    *Multi        |  1    |    74.81  |   x     |    x     |      x  |
| EfficientDet-D3 | EfficientNet-B3 |    *Multi        |  1    |    74.53  |   x     |    x     |      x  |

* `*` results achived with frozeen weights from detection task (after addition of mask head)
* `**` `Multi` - name for mixed open dataset (with listed below datasets) for detection/segmentation task


### Classification task

```Under construction - TBA```

## Project Organization (WIP)
------------

    ├── LICENSE
    ├── README.md 
    |         <- The top-level README for developers using this project.
    ├── docs               <- documents
    │
    ├── logs          	    <- Tracking experiments e.g. Tensorboard
    |
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.


--------