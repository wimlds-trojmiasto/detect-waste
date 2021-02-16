'''
Script to sort openlittermap images for 7-class litter classification.
'''
import argparse
import json
import os
from shutil import copy2
import sys
sys.path.append('..')

from pycocotools.coco import COCO
from tqdm import tqdm

from cut_bbox_litter import crop
from utils.dataset_converter import label_to_detectwaste


def extract_category(ann):
    return ann['properties']['result_string'].split('.')[1].split(' ')[0]


def extract_filename(ann):
    return str(ann['properties']['photo_id']) + '.'
    + ann['properties']['filename'].split('.')[-1]


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Prepare images of trash for classification task')
    parser.add_argument('--src_ann', help='path to openlittermap annotations',
                        default='/dih4/dih4_2/wimlds/zklawikowska/openlittermap/jsondata.json',
                        type=str)
    parser.add_argument('--coco', default='/dih4/dih4_home/smajchrowska/openlittermap.json',
                        help='path to our coco openlittermap annotations',
                        type=str)
    parser.add_argument('--src_img',
                        help='path to source directory with images',
                        type=str, default='/dih4/dih4_2/wimlds/zklawikowska/openlittermap/images/')
    parser.add_argument('--dst_img',
                        help='path to destination directory for images',
                        type=str, default='/dih4/dih4_2/wimlds/smajchrowska/openlittermap/')
    return parser


def main(args):
    print(args)
    with open(args.src_ann, 'r') as f:
        dataset = json.load(f)
    if args.coco:
        with open(args.coco, 'r') as f:
            coco_dataset = json.load(f)
        coco = COCO(args.coco)

        # build a dictionary mapping the file name to the image id
        images_map = {}
        for img_obj in coco_dataset['images']:
            file_name = img_obj['file_name']
            id = img_obj['id']
            images_map[file_name] = id

    anns = dataset['features']
    for ann in tqdm(anns):
        try:
            category = extract_category(ann)
            label = label_to_detectwaste(category)
            img_src = os.path.join(args.src_img, extract_filename(ann))
            img_path = os.path.join(args.dst_img, label, extract_filename(ann))
            if not args.coco:
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                copy2(img_src, img_path)
            else:
                img_id = images_map[extract_filename(ann)]
                annIds = coco.getAnnIds(imgIds=img_id)
                coco_anns = coco.loadAnns(annIds)
                for coco_a in coco_anns:
                    crop(coco_a, 'pseudolabel', extract_filename(ann),
                         label, True, 1,
                         args.src_img, args.dst_img, i=coco_a['id'])
        except:
            continue


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
