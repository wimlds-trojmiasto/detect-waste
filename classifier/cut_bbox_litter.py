'''
Script to prepare images for litter classification.
'''
import sys
sys.path.append('..')
import os
import json
import cv2
import argparse
import glob
from tqdm import tqdm
from utils.dataset_converter import convert_categories_to_detectwaste
                                        
def get_args_parser():
    parser = argparse.ArgumentParser(
        'Prepare images of trash for classification task')
    parser.add_argument('--src_coco', help='path to coco annotations',
                        type=str)
    parser.add_argument('--src_img',
                        help='path to source directory with images',
                        type=str, default='/dih4/dih4_2/wimlds/data/all_detect_images')
    parser.add_argument('--dst_img',
                        help='path to destination directory for images',
                        type=str, default='images_square/')
    parser.add_argument('--name',
                        help='type of annotations: wimlds or epi',
                        default='wimlds',
                        choices=['wimlds', 'epi'],
                        type=str)
    return parser

def main(args):
    print(args)
    modes =['train', 'test']
    cut_square = True
    
    for mode in modes:
        print("Running for mode ", mode)
        ann_list = []
        path = '../annotations/*_*_'+mode+".json"
        for file in glob.glob(path):
            ann_list.append(file)
        if len(ann_list) < 1:
            print("No annotations detected")
            
        if not os.path.exists(os.path.join(args.dst_img, mode)):
            os.makedirs(os.path.join(args.dst_img, mode))
            
        for i, train_path in enumerate(ann_list):
            # first, move all category ids to detectwaste (7 categories)
            print("Converting file ", train_path)
            data_all = convert_categories_to_detectwaste(source=train_path,
                                        dest=None)
            wimlds_list = data_all['categories']
            
            print(wimlds_list)
            mapping_category = {}
            for item in wimlds_list:
                mapping_category[item['id']] = item['name']
                if not os.path.exists(os.path.join(args.dst_img, mode, item['name'])):
                    os.mkdir(os.path.join(args.dst_img, mode, item['name']))

            # build a dictionary mapping the image id to the file name
            images = {}
            for img_obj in data_all['images']:
                file_name = img_obj['file_name']
                id = img_obj['id']
                images[id] = file_name

            for annotation_obj in tqdm(data_all['annotations']):
                # read information from 'annotations'
                annotation_id = str(i) + str(annotation_obj['id'])
                image_id = int(annotation_obj['image_id'])
                file_name = os.path.join(args.src_img, images[image_id])

                category_name = mapping_category[annotation_obj['category_id']]
                # prepare for cropping - USING THE BBOX's
                # WIDTH AND HEIGHT HERE
                x, y, width, height = annotation_obj['bbox']
                img = cv2.imread(file_name)
                if cut_square:
                    if width > height:
                        x = x - (width-height)/2
                        height = width
                    else:
                        y = y - (-width+height)/2
                        width = height
                    
                # somehow some coordinates are negative???
                crop_img = img[int(abs(y)): int(abs(y) + abs(height)),
                            int(abs(x)): int(abs(x) + abs(width))]
                
                
                    
                try:
                    cv2.imwrite(os.path.join(args.dst_img, mode, category_name,
                                            annotation_id +'.jpg'),
                                crop_img)
                except BaseException:
                    print(f"ERROR: {file_name}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
