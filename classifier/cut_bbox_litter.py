'''
Script to prepare images for litter classification.
'''
import os
import json
import cv2
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Prepare images of trash for classification task')
    parser.add_argument('--src_coco', help='path to coco annotations',
                        type=str)
    parser.add_argument('--src_img',
                        help='path to source directory with images',
                        type=str)
    parser.add_argument('--dst_img',
                        help='path to destination directory for images',
                        type=str)
    parser.add_argument('--name',
                        help='type of annotations: wimlds or epi',
                        default='wimlds',
                        choices=['wimlds', 'epi'],
                        type=str)
    return parser


epi_map = {1: 4,
           2: 0,
           3: 2,
           4: 6,
           5: 1,
           6: 3,
           7: 5}


def ChangeIdFunction(id):
    glass_id = [6, 9, 26]
    metals_and_plastics_id = [0, 4, 5, 7, 8, 10, 11, 12, 16, 21, 24,
                              27, 28, 29, 36, 37, 40, 41, 42, 43, 44,
                              45, 47, 48, 49, 50, 52, 55]
    non_recyclable_id = [2, 3, 18, 19, 20, 22, 23, 30, 31,
                         32, 33, 35, 38, 39, 46, 51, 53, 54, 56, 57, 59]
    other_id = [1]
    paper_id = [13, 14, 15, 17, 33, 34]
    bio_id = [25]
    unlabeld_id = [58]

    if (id in glass_id):
        id = 0
        return id
    if (id in metals_and_plastics_id):
        id = 1
        return id
    if (id in non_recyclable_id):
        id = 2
        return id
    if (id in other_id):
        id = 3
        return id
    if (id in paper_id):
        id = 4
        return id
    if (id in bio_id):
        id = 5
        return id
    if (id in unlabeld_id):
        id = 6
        return id
    else:
        print("STH wrong")
        id = 7
        return id


def main(args):
    print(args)
    wimlds_list = [{'supercategory': 'Litter', 'id': 0, 'name': 'Glass'},
                   {'supercategory': 'Litter', 'id': 1,
                    'name': 'Metals-and-plastics'},
                   {'supercategory': 'Litter',
                    'id': 2, 'name': 'Non-recyclable'},
                   {'supercategory': 'Litter', 'id': 3, 'name': 'Other'},
                   {'supercategory': 'Litter', 'id': 4, 'name': 'Paper'},
                   {'supercategory': 'Litter', 'id': 5, 'name': 'Bio'},
                   {'supercategory': 'Litter', 'id': 6, 'name': 'Unknown'}
                   ]

    with open(args.src_coco, 'r') as f:
        data_all = json.load(f)

    data_all['categories'] = wimlds_list

    if not os.path.exists(args.dst_img):
        os.mkdir(args.dst_img)
    mapping_category = {}
    for item in wimlds_list:
        mapping_category[item['id']] = item['name']
        if not os.path.exists(os.path.join(args.dst_img, item['name'])):
            os.mkdir(os.path.join(args.dst_img, item['name']))

    # build a dictionary mapping the image id to the file name
    images = {}
    for img_obj in data_all['images']:
        file_name = img_obj['file_name']
        id = img_obj['id']
        images[id] = file_name

    for annotation_obj in data_all['annotations']:
        # take our name for trash
        if args.name == 'epi':
            annotation_obj['category_id'] = \
                epi_map[annotation_obj['category_id']]
        else:
            annotation_obj['category_id'] = ChangeIdFunction(
                annotation_obj['category_id'])
        # get the bounding box of each annotation in the image
        # and crop it

        # read information from 'annotations'
        annotation_id = str(annotation_obj['id'])
        image_id = int(annotation_obj['image_id'])
        file_name = os.path.join(args.src_img, images[image_id])

        category_name = mapping_category[annotation_obj['category_id']]
        # prepare for cropping - USING THE BBOX's
        # WIDTH AND HEIGHT HERE
        x, y, width, height = annotation_obj['bbox']
        img = cv2.imread(file_name)
        # somehow some coordinates are negative???
        crop_img = img[int(abs(y)): int(abs(y) + abs(height)),
                       int(abs(x)): int(abs(x) + abs(width))]
        try:
            cv2.imwrite(os.path.join(args.dst_img, category_name,
                                     annotation_id + '.jpg'),
                        crop_img)
        except BaseException:
            print(f"ERROR: {file_name}, {crop_img}, "
                  f"{annotation_obj['bbox']}, {annotation_obj['image_id']}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if not os.path.exists(args.dst_img):
        os.mkdir(args.dst_img)
    main(args)
