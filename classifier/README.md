# CNN (PyTorch) based on ResNet50, and EfficientNet (PyTorch Lightning) for waste classification

Many objects in TACO dataset have Unknown label. This waste is mostly invisible or destroyed.
To address this challenge at the early stage of the project we trained classifier to this type of waste to know their true category.

Additionally during our project we realized that existing datasets do not provide a large number of object classes with sufficient annotated training data. In addition, as we managed to find out, differentiating waste instances under a single class label is also challenging. In this regard, we decided to formulate our problem as a one-class object detection, and classification in next step.

# Implementation
A PyTorch script for litter classification:
 1) `ResNet` - based on implementation from [Tutorial on training ResNet](https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5?gi=ecba7eb12775),
 2) `EfficientNet` - EfficientNet implementation with pseudo-labeling technique. (implemented using [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning))

Additionally to address class imbalance we used `WeightedRandomSampler`.

## Requirements
` pip install -r requirements.txt `

## Neptune
To track logs (for example training loss) we used [neptune.ai](https://neptune.ai/). If you are interested in logging your experiments there, you should create account on the platform and create new project. Then:
* Find and set Neptune API token on your system as environment variable (your NEPTUNE_API_TOKEN should be added to ~./bashrc)
* Add your project_qualified_name name in the `train_<net_name>.py`
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
* we used TACO dataset with additional annotated data from detect-waste,
* we used few waste detection/segmentation dataset mentioned in main `README.md`,
* we used [TrashNet](https://github.com/garythung/trashnet) and [waste_pictures](https://www.kaggle.com/wangziang/waste-pictures), and some pictures collected using [Google Images Download](https://github.com/hardikvasa/google-images-download).

We expect the images directory structure to be the following:
```
path/to/images/          # all images
  images_square/         
    pseudolabel/         # unlabeled data used in pseudo-labeling task
    test/                # images divided into categories - test subset
      background/
      bio/
      glass/
      metals_and_plastic/
      non_recyclable/
      other/
      paper/
      unknown/
    train/                # images divided into categories - train subset
      background/
      bio/
      glass/
      metals_and_plastic/
      non_recyclable/
      other/
      paper/
      unknown/
```

# Models

* ## Modified ResNet50

    Backbone of classificator is ResNet50 taken from torchvision storage.

    ### To run test
    ```bash
    python train_resnet.py --data_img path/to/images/images_square/test/ \
                           --out path/to/checkpoints/ \
                           --mode test \
                           --name test.jpg \
                           --device cpu \
    ```

    ### To run training (on GPU id=1)
    ```bash
    python train_resnet.py --data_img path/to/images/train/ \
                           --out path/to/checkpoints/ \
                           --mode train \
                           --device cuda:1 \
    ```

* ## EfficientNet

    This implementation uses [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) implementation.
    EfficientNet is implemented with PyTorch Lightning.

    ### To run training use `train_effnet.py`

    ```bash
    python train_effnet.py --data_img path/to/images/train/ \
                           --save path/to/checkpoint.ckpt \
                           --model efficientnet-b2 \
                           --gpu 1 \
                           --pseudolabel_mode per-batch \
                           --neptune \
    ```

    * efficientnet - any efficientnet form b0 to b7 can be used
    * pseudolabeling - allow to use unannotated data. Can be use in `per-epoch` and `per-batch` modes, which refers to how often    pseudolabels will be upadted
    * data augmentation - any data augmentation form  [albumentations](https://github.com/albumentations-team/albumentations) can be applied



# Performance

|      model      | # classes | ACC | sampler | pseudolabeling |
| :--------------:| :-------: | :--:| :-----: | :------------: |
| EfficientNet-B2 | 8         |73.02| Weighted| per batch      |
| EfficientNet-B2 | 8         |74.61| Random  | per epoch      |
| EfficientNet-B2 | 8         |72.84| Weighted| per epoch      |
| EfficientNet-B4 | 7         |71.02| Random  | per epoch      |
| EfficientNet-B4 | 7         |67.62| Weighted| per epoch      |
| EfficientNet-B2 | 7         |72.66| Random  | per epoch      |
| EfficientNet-B2 | 7         |68.31| Weighted| per epoch      |
| EfficientNet-B2 | 7         |74.43| Random  | None           |
| ResNet-50       | 8         |60.60| Weighted| None           |

* 8 classes - 8th class for additional background category
* we provided 2 methods to update pseudo-labels: per batch and per epoch
