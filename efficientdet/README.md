# EfficientDet (PyTorch) for waste detection

# Implementation
A PyTorch EfficientDet for detecting waste using implementation from  [efficientdet-pytorch by rwightman](https://github.com/rwightman/efficientdet-pytorch)

# Work in progress
* simplify repo 1: move to pytorch-lightning (simplifying pipeline)
* simplify repo 2: remove unnecessary code and files
* add data augmentations methods from albumentations library
* add advanced data augmentation
* use unannotated data

# Environment
### Requirements
` pip install -r requirements.txt `

### Neptune
* Find and set Neptune API token on your system as environment variable (your NEPTUNE_API_TOKEN should be added to ~./bashrc)
* Add your project_qualified_name name in the `train.py`
    `neptune.init(project_qualified_name = 'YOUR_PROJECT_NAME/detect-waste') `

    Currently it is set to private detect-waste neptune space.

# Data
### Dataset
* we use TACO dataset with additional annotated data from detect-waste (WIP)

### Data preprocessing
* go to *detect-waste* repository
* run *annotations_preprocessing.py*

    ` python3 annotations_preprocessing.py `

    new annotations will be saved in *annotations/annotations_train.json* and *annotations/annotations_test.json*

### Dataset config
* dataset config should be set in *efficientdet/effdet/data/dataset_config.py*

    Remember to change paths to your data. Now it is set to work with currently running project on dih4.ai

    `class DetectwasteCfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        val=dict(ann_filename=PATH_TO_YOUR_JSON_VAL_ANNOTATIONS, img_dir=PATH_TO_YOUR_VAL_DATA, has_labels=True),
        train=dict(ann_filename=PATH_TO_YOUR_JSON_TRAIN_ANNOTATIONS, img_dir=PATH_TO_YOUR_TRAIN_DATA, has_labels=True),
    ))`


# Training

### Run training
Example run:

* simply run `bash efficientdet/train.sh`

    Example configuration:

    `python3 efficientdet/train.py "/dih4/dih4_2/wimlds/data/" \
        --model tf_efficientdet_d2 --batch-size 4 --decay-rate 0.95 \
        --lr .001 --workers 4 --warmup-epochs 5 --model-ema --dataset DetectwasteCfg \
        --pretrained --num-classes 7 --color-jitter 0.1 --reprob 0.2 --epochs 200 `

### Training customization

* architecture

    All avaiable architectures can be found in `efficientdet/effdet/config/model_config.py`
        
    To use other architecture from *model_config.py* select a dict and use it as `--model` param in your  `efficientdet/train.sh` file e.g. `mobiledetv3_large`


    `python3 efficientdet/train.py "/dih4/dih4_2/wimlds/data/" \
    --model mobiledetv3_large --batch-size 4 --decay-rate 0.95 \
    --lr .001 --workers 4 --warmup-epochs 5 --model-ema --dataset DetectwasteCfg \
    --pretrained --num-classes 7 --color-jitter 0.1 --reprob 0.2 --epochs 200 `

* selecting GPUs (WIP)
* selecting data augmentation methods (WIP)
