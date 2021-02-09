
'''
Script to sort oepenlittermap images for 7-class litter classification.
'''
import sys
import argparse
sys.path.append('..')
import os
import json
from tqdm import tqdm
from shutil import copy2
from utils.dataset_converter import label_to_detectwaste
    
def extract_category(ann):
    return ann['properties']['result_string'].split('.')[1].split(' ')[0]
def extract_filename(ann):
    return str(ann['properties']['photo_id']) +'.'+ ann['properties']['filename'].split('.')[-1]
                      
def get_args_parser():
    parser = argparse.ArgumentParser(
        'Prepare images of trash for classification task')
    parser.add_argument('--src_ann', help='path to openlittermap annotations',
                        type=str)
    parser.add_argument('--src_img',
                        help='path to source directory with images',
                        type=str, default='/dih4/dih4_2/wimlds/zklawikowska/openlittermap/drive/')
    parser.add_argument('--dst_img',
                        help='path to destination directory for images',
                        type=str, default='/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/classifier/openlittermap/)
    return parser

def main(args):
    print(args)
    with open(args.src_ann, 'r') as f:
            dataset = json.loads(f.read())

    anns = dataset['features']
    categories = []
    for ann in tqdm(anns):
        try:
            category = extract_category(ann)
            label = label_to_detectwaste(category)
            img_src = os.path.join(args.src_img,extract_filename(ann))
            img_path = os.path.join(args.dst_img, label, extract_filename(ann))
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            copy2(img_src,img_path)
        except:
            continue


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
