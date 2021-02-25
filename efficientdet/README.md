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
    python3 efficientdet/train.py /dih4/dih4_2/wimlds/data/all_detect_images \
        --ann_name annotations/annotations --model tf_efficientdet_d2 \
        --batch-size 4 --decay-rate 0.95 --lr .001 --workers 4 --warmup-epochs 5 \
        --model-ema --dataset detectwaste --pretrained --num-classes 7 \
        --color-jitter 0.1 --reprob 0.2 --epochs 20 --device cuda:1
    ```

    For single class in mixed dataset:

    ```bash
    python3 efficientdet/train.py /dih4/dih4_2/wimlds/data/all_detect_images \
    --ann_name annotations/binary_mixed --model tf_efficientdet_d2 \
    --batch-size 4 --decay-rate 0.95 --lr .001 --workers 4 --warmup-epochs 5 \
    --model-ema --dataset multi --pretrained --num-classes 1 --color-jitter 0.1 \
    --reprob 0.2 --epochs 20 --device cuda:1
    ```

### Training customization

* architecture

    All avaiable architectures can be found in `efficientdet/effdet/config/model_config.py`
        
    To use other architecture from *model_config.py* select a dict and use it as `--model` param in your  `efficientdet/tools/train.sh` file e.g. `mobiledetv3_large`

    ```bash
    python3 efficientdet/train.py /dih4/dih4_2/wimlds/data/all_detect_images \
    --ann_name annotations/annotations --model mobiledetv3_large \
    --batch-size 4 --decay-rate 0.95 --lr .001 --workers 4 --warmup-epochs 5 \
    --model-ema --dataset detectwaste --pretrained --num-classes 7 \
    --color-jitter 0.1 --reprob 0.2 --epochs 200
    ```

* selecting GPUs provide desirable gpu id: `--device cuda:1`
* selecting data augmentation methods (WIP)

# Evaluation

We provided `demo.py` script to draw bounding boxes on choosen image. For example script can be run on GPU (id=0) with arguments:
```bash
    python demo.py --save path/to/save/image.png --checkpoint path/to/checkpoint.pth \
                   --img path/or/url/to/image --device cuda:0
```
or on video with `--video` argument:
```bash
    python demo.py --save directory/to/save/frames --checkpoint path/to/checkpoint.pth \
                   --img path/to/video.mp4 --device cuda:0 --video \
                   --classes label0 label1 label2
```

If you managed to process all the frames, just run the following command from the directory where you saved the results:
```bash
    ffmpeg -i img%08d.jpg movie.mp4
```
### Run Evaluation
Example evaluation run with COCO metrics:

```bash
    python3 efficientdet/validate.py "/dih4/dih4_2/wimlds/data/all_detect_images" \
    --ann_name "../annotations/binary_mixed" --model tf_efficientdet_d2 \
    --split val --batch-size 4 --workers 4 --checkpoint "/path/to/checkpoint.pth.tar" \
    --dataset multi --num-classes 1
```
### Create Pseudolabels
There is also a posibility to create pseudolabels in coco format by runing:
```bash
    python pseudolabel.py --dst_coco /path/to/save/coco.json --src_img /path/to/images \
                          --dst_images /optional/directory/to/save/images/with/bboxes \
                          --checkpoint path/to/checkpoint.pth.tar --score-thr 0.3 \
                          --device cuda:0 --classes label0 label1 label2
```

# Performance

### 7-class TACO bboxes

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.119
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.162
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.130
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.064
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.094
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.216
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.406
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.234
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.339
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.475
```

### 1-class TACO bboxes

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.568
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.435
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.198
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692
```

### Mixed datasets

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.443
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.640
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.055
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.475
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.561
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.605
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.410
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689
```
