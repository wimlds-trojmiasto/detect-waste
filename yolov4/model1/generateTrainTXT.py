import sys
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

import taco_to_dataset

module_path = str(Path.cwd().parents[0] / "src")
if module_path not in sys.path:
    sys.path.append(module_path)


dataset_path = '/dih4/dih4_2/wimlds/TACO-master/data'
anns_file_path = dataset_path + '/' + 'annotations.json'
epinote_dataset_path = '/dih4/dih4_2/wimlds/data/not-annotated'
epinote_anns_file_path = '/dih4/dih4_2/wimlds/data/annotations_epi.json'
# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())
with open(epinote_anns_file_path, 'r') as f:
    epinote_dataset = json.loads(f.read())

for cat in epinote_dataset['categories']:
    new_id = taco_to_dataset.epi_to_detectwaste(cat['id'])
    cat['id'] = new_id
for ann in epinote_dataset['annotations']:
    new_cat = taco_to_dataset.epi_to_detectwaste(ann['category_id'])
    ann['category_id'] = new_cat

def get_dataset():
    return  dataset

dataset = get_dataset()
imgs = dataset['images']
anns = dataset['annotations']
print(imgs[0])

f = open("train.txt", "w")
for img in imgs:
    file_name = img['file_name']
    path = '/dih4/dih4_2/wimlds/TACO-master/data/'+file_name
    objects = ''
    for ann in anns:
        if ann['image_id'] == img['id']:
            x1 = ann['bbox'][0]
            y1 = ann['bbox'][1]
            x2 = ann['bbox'][0]+ann['bbox'][2]
            y2 = ann['bbox'][0]+ann['bbox'][3]
            label = ann['category_id']
            objects = objects+'%d,%d,%d,%d,%d ' %(x1,y1,x2,y2,label)
    annotatedImage = path+' '+objects+'\n'
    f.write(annotatedImage)