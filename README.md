# Detect waste
AI4Good project for detecting waste in environment
www.detectwaste.ml

## Data download (WIP)
* TACO bboxes - in progress. TACO dataset can be downloaded (http://tacodataset.org/)[here]. TACO bboxes will be avaiable for download soon.

## Data preprocessing
* run *annotations_preprocessing.py*

    `python3 annotations_preprocessing.py`

    new annotations will be saved in *annotations/annotations_train.json* and *annotations/annotations_test.json*
# Models

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