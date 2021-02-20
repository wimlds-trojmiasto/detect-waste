# EfficientDet for waste detection (WIP)

This repository is still at early stage of development.

# Implementation (PyTorch)
A PyTorch EfficientDet for detecting waste using implementation from  [efficientdet-pytorch by rwightman](https://github.com/rwightman/efficientdet-pytorch)

# Work in progress
* simplify repo 1: move to pytorch-lightning (simplifying pipeline)
* simplify repo 2: remove unnecessary code and files
* add data augmentations methods from albumentations library
* add advanced data augmentation
* use unannotated data

# Environment
### Requirements
` pip install -r requirements/requirements.txt `

### Neptune
To track logs (for example training loss) we used [neptune.ai](https://neptune.ai/). If you are interested in logging your experiments there, you should create account on the platform and create new project. Then:
* Find and set Neptune API token on your system as environment variable (your NEPTUNE_API_TOKEN should be added to ~./bashrc)
* Add your project_qualified_name name in the `train.py`
    ```python
      neptune.init(project_qualified_name = 'YOUR_PROJECT_NAME/detect-waste')
    ```
    Currently it is set to private detect-waste neptune space.
* install neptun-client library
    ```bash
      pip install neptune-client
    ```

To run experiments with neptune simply add `--neptune` flag during launch `train.py`.

For more check [LINK](https://neptune.ai/how-it-works).

# Dataset
* we use TACO dataset with additional annotated data from detect-waste,
* we use few waste dataset mentioned in main `README.md` with annotated data by bbox (and sometimes also with mask).

We expect the directory structure to be the following:
```
path/to/repository/
  annotations/         # annotation json files
path/to/images/        # all images
```
You can modify `effdet/data/dataset_config.py` and `effdet/data/dataset_factory.py` dataset classes to add new dataset and another format of paths for coco annotations type.

Check `detect-waste/annotations/README.md` to verify provided annotations by [Detect Waste in Pomerania team](https://detectwaste.ml/).

### Data preprocessing
* go to *detect-waste* repository
* run *annotations_preprocessing.py*
    ```python
    python3 annotations_preprocessing.py
    ```
    * new annotations for 7-classes case will be saved in `annotations/annotations_train.json` and `annotations/annotations_test.json`, and additionaly,
    * new annotations for 1-class case will be saved in `annotations/annotations_binary_train.json` and `annotations/annotations_binary_test.json`.
* or just use annotations stored at `detect-waste/annotations/` directory.

### Dataset config
* dataset config should be set in *efficientdet/effdet/data/dataset_config.py*

    Remember to provide paths to your annotations during runing a main script `train.py`.

    ```python
    class DetectwasteCfg(CocoCfg):
        root: str = ""
        ann: str = ""
        variant: str = '2017'
        num_classes: int = 7

        def add_split(self):
            self.splits = {
                'train': {'ann_filename': self.ann+'_train.json',
                        'img_dir': self.root,
                        'has_labels': True},
                'val': {'ann_filename': self.ann+'_test.json',
                        'img_dir': self.root,
                        'has_labels': True}
                }
    ```

    If you plan to train on single class remember to use single class configuration.

# Training

### Run training
Example run:

* simply run `bash efficientdet/tools/train.sh`

    Example configuration for running on gpu with id=1:

    ```bash
    python3 efficientdet/train.py "/dih4/dih4_2/wimlds/data/all_detect_images" \
        --ann_name "../annotations/annotations_" --model tf_efficientdet_d2 \
        --batch-size 4 --decay-rate 0.95 --lr .001 --workers 4 --warmup-epochs 5 \
        --model-ema --dataset DetectwasteCfg --pretrained --num-classes 7 \
        --color-jitter 0.1 --reprob 0.2 --epochs 20 --device cuda:1
    ```

    For single class in mixed dataset:

    ```bash
    python3 efficientdet/train.py "/dih4/dih4_2/wimlds/data/all_detect_images" \
    --ann_name "../annotations/binary_mixed_" --model tf_efficientdet_d2 \
    --batch-size 4 --decay-rate 0.95 --lr .001 --workers 4 --warmup-epochs 5 \
    --model-ema --dataset multi --pretrained --num-classes 1 --color-jitter 0.1 \
    --reprob 0.2 --epochs 20 --device cuda:1
    ```

### Training customization

* architecture

    All avaiable architectures can be found in `efficientdet/effdet/config/model_config.py`
        
    To use other architecture from *model_config.py* select a dict and use it as `--model` param in your  `efficientdet/tools/train.sh` file e.g. `mobiledetv3_large`

    ```bash
    python3 efficientdet/train.py "/dih4/dih4_2/wimlds/data/all_detect_images" \
    --ann_name "../annotations/annotations_" --model mobiledetv3_large \
    --batch-size 4 --decay-rate 0.95 --lr .001 --workers 4 --warmup-epochs 5 \
    --model-ema --dataset DetectwasteCfg --pretrained --num-classes 7 \
    --color-jitter 0.1 --reprob 0.2 --epochs 200
    ```

* selecting GPUs provide desirable gpu id: `--device cuda:1`
* selecting data augmentation methods (WIP)
