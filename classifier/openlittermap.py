
'''
Script to prepare images for litter classification.
'''
import sys
sys.path.append('..')
import os
import json
from tqdm import tqdm
from shutil import copy2
from utils.dataset_converter import label_to_detectwaste

source = '/dih4/dih4_2/wimlds/zklawikowska/openlittermap/jsondata.json'
dest = '/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/classifier/openlittermap/'
images = '/dih4/dih4_2/wimlds/zklawikowska/openlittermap/drive/'
     
def extract_category(ann):
    return ann['properties']['result_string'].split('.')[1].split(' ')[0]
def extract_filename(ann):
    return str(ann['properties']['photo_id']) +'.'+ ann['properties']['filename'].split('.')[-1]                      

with open(source, 'r') as f:
        dataset = json.loads(f.read())

anns = dataset['features']
categories = []
for ann in tqdm(anns):
    try:
        category = extract_category(ann)
        label = label_to_detectwaste(category)
        img_src = os.path.join(images,extract_filename(ann))
        img_dest = os.path.join(dest, label, extract_filename(ann))
        os.makedirs(os.path.dirname(img_dest), exist_ok=True)
        copy2(img_src,img_dest)
    except:
        continue
        