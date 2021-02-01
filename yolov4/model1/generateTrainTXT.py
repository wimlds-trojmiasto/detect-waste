import sys
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


from PIL import Image, ExifTags
from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
import random
import pylab

from utils.taco_to_detectwaste import epi_to_detectwaste

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset_path', default='/dih4/dih4_2/wimlds/TACO-master/data', type=str, metavar='PATH',
                    help='Dataset path (default: /dih4/dih4_2/wimlds/TACO-master/data)')
parser.add_argument('--anns_file_path', default='/dih4/dih4_2/wimlds/TACO-master/data/annotations.json', type=str, metavar='PATH',
                    help='Annotations path (default: /dih4/dih4_2/wimlds/TACO-master/data/annotations.json)')
parser.add_argument('--epinote_dataset_path', default='/dih4/dih4_2/wimlds/data/not-annotated', type=str, metavar='PATH',
                    help='Epinote dataset path (default: /dih4/dih4_2/wimlds/data/not-annotated)')
parser.add_argument('--epinote_anns_file_path', default='/dih4/dih4_2/wimlds/data/annotations_epi.json', type=str, metavar='PATH',
                    help='Epinote annotations path (default: /dih4/dih4_2/wimlds/data/annotations_epi.json)')
module_path = str(Path.cwd().parents[0] / "src")
if module_path not in sys.path:
    sys.path.append(module_path)

args = parser.parse_args()
# Read annotations
with open(args.dataset_path, 'r') as f:
    dataset = json.loads(f.read())
with open(args.epinote_anns_file_path, 'r') as f:
    epinote_dataset = json.loads(f.read())

for cat in epinote_dataset['categories']:
    new_id = epi_to_detectwaste(cat['id'])
    cat['id'] = new_id
for ann in epinote_dataset['annotations']:
    new_cat = epi_to_detectwaste(ann['category_id'])
    ann['category_id'] = new_cat

def get_dataset():
    return  epinote_dataset

dataset = get_dataset()
imgs = dataset['images']
anns = dataset['annotations']
labels=[]
print(imgs[0])

#f = open("train.txt", "w")
for img in imgs:
    file_name = img['file_name']
    path = file_name
    image = Image.open(args.dataset_path + path)
    print(image)
    objects = ''
    for ann in anns:
        if ann['image_id'] == img['id']:
            x1 = ann['bbox'][0]
            y1 = ann['bbox'][1]
            x2 = ann['bbox'][0]+ann['bbox'][2]
            y2 = ann['bbox'][0]+ann['bbox'][3]
            label = ann['category_id']
            objects = objects+' %d,%d,%d,%d,%d' %(x1,y1,x2,y2,label)
            labels.append(label)
    annotatedImage = path+objects+'\n'

print(max(labels))
print(min(labels))
print(labels.count(0))
print(labels.count(1))
print(labels.count(2))
print(labels.count(3))
print(labels.count(4))
print(labels.count(5))
print(labels.count(6))
print(len(labels))
    #f.write(annotatedImage)