# CNN (PyTorch) based on ResNet50 for waste classification

# Problem
Many object in TACO dataset has Unknown label.
This waste is is mostly invisible or destroyed.
To adress this challenge we tried to train classificator tor this type of waste.

# Implementation
A PyTorch script for litter classification based on implementation from [Tutorial on training ResNet](https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5?gi=ecba7eb12775)
Additionaly to address class imbalance we used WeightedRandomSampler.

# Dataset
* we use TACO dataset with additional annotated data from detect-waste
* backbone of classificator is ResNet50 from torchvision storage
* data for training can be found at ```/dih4/dih4_2/wimlds/smajchrowska/categories/```

# Usage

```python run.py --data_img /dih4/dih4_2/wimlds/smajchrowska/categories/train/ --out /dih4/dih4_2/wimlds/smajchrowska/epi_categories --mode test --name test.jpg --device cpu```

```python run.py --data_img /dih4/dih4_2/wimlds/smajchrowska/categories/train/ --out /dih4/dih4_2/wimlds/smajchrowska/categories --mode train --device cpu```
