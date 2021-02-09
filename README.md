# Detect waste
AI4Good project for detecting waste in environment
www.detectwaste.ml

## Results

### Detection/Segmentation task
| model  | backbone  | Dataset       | # classes | bbox AP@0.5 | bbox AP@0.5:0.95 | mask AP@0.5 | mask AP@0.5:0.95 |
| :-----:| :-------: | :-----------: | :-------: | :---------: | :--------------: | :---------: | :--------------: |
| DETR    | ResNet 50 |   TACO bboxes | 1        |    42.72    |       20.66      |      x      |  x               |
| DETR    | ResNet 50 |   TACO bboxes | 7        |    6.17     |       3.03       |      x      |  x               |
| DETR    | ResNet 50 |   Multi       | 1        |    37.93    |       19.43      |      x      |  x               |
| EfficientDet-D2 | EfficientNet-B2 |    Taco bboxes  |  1    |    61.05  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    Taco bboxes  |  7    |    18.78  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    Drink-waste  |  4    |    99.60  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    MJU-Waste    |  1    |    97.74  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    TrashCan v1  |  8    |    91.28  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    Wade-AI      |  1    |    33.03  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    UAVVaste     |  1    |    79.90  |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    Trash ICRA19 |  7    |    9.47   |   x     |    x     |      x  |
| EfficientDet-D2 | EfficientNet-B2 |    Multi        |  1    |    74.81  |   x     |    x     |      x  |
| EfficientDet-D3 | EfficientNet-B3 |    Multi        |  1    |    74.53  |   x     |    x     |      x  |
| Mask R-CNN  | ResNet 50    |  Multi   |  1    |    27.95 |       16.49   |    23.05     |    12.94       |
| Mask R-CNN  | ResNetXt 101 |  Multi   |  1    |    19.70 |       6.20    |    24.70     |    13.20       |


* `Multi` - name for mixed open dataset (with listed below datasets) for detection/segmentation task

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

To read more about waste detection check [litter-detection-review](https://github.com/majsylw/litter-detection-review)

* ### Efficientdet (WIP)

    To train efficientdet check `efficientdet/README.md`

* ### DETR

    To train detr check `detr/README.md` (WIP)

    PyTorch training code and pretrained models for **DETR** (**DE**tection **TR**ansformer).
    We replace the full complex hand-crafted object detection pipeline with a Transformer, and match Faster R-CNN with a ResNet-50, obtaining **42 AP** on COCO using half the computation power (FLOPs) and the same number of parameters. Inference in 50 lines of PyTorch.

    **What it is**. Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. 
    Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

    For details see [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko.

* ### FastRCNN
    To train FastRCNN check `FastRCNN/README.md`

* ### ResNet50 (only classification)
   To train ResNet50 check `classifier/README.md`


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