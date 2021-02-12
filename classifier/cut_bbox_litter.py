'''
Script to prepare images for litter classification.
'''
from multiprocessing import Process, Manager
from typing import List, Dict, Generator, Any

import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(),
                             os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import cv2
import argparse
import glob
from tqdm import tqdm
from utils import convert_categories_to_detectwaste


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Prepare images of trash for classification task')
    parser.add_argument('--src_coco',
                        help='path to directory with coco annotations',
                        type=str,
                        default='../annotations')
    parser.add_argument('--src_img',
                        help='path to source directory with images',
                        type=str,
                        default='/dih4/dih4_2/wimlds/data/all_detect_images')
    parser.add_argument('--dst_img',
                        help='path to destination directory for images',
                        type=str, default='images_square/')
    parser.add_argument('--jobs',
                        help='number of multiprocess to run',
                        type=int, default=50)
    # * square shape
    parser.add_argument('--square', action='store_true',
                        help="cut images into square shape")
    # zoom. useful for classification when used witg
    # detection algorithm that select not bbox coordinates
    # however can lower the scores if images are
    # crowded with many objects
    parser.add_argument('--zoom',
                        help='zoom out or in bounding box',
                        type=int, default=1.2)
    return parser


def crop(annotations_list,
         i, mode, images, mapping_category,
         src_img, dst_img, zoom, cut_square=True):
    for annotation_obj in tqdm(annotations_list):
        # read information from 'annotations'
        annotation_id = str(i) + str(annotation_obj['id'])
        image_id = int(annotation_obj['image_id'])
        file_name = os.path.join(src_img, images[image_id])

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
        width *= zoom
        height *= zoom
        crop_img = img[int(y): int(y + height),
                       int(x): int(x + width)]

        try:
            cv2.imwrite(os.path.join(args.dst_img, mode, category_name,
                                     annotation_id + '.jpg'),
                        crop_img)
        except BaseException:
            print(f"ERROR: {file_name}, img id: {image_id}")


def split_list(files: List[Dict],
               no_of_jobs: int
               ) -> Generator[List[str], Any, None]:
    ''' Divide list into no_of_jobs sublists
    (images in each list will be
    processed by different process)
    '''
    k, m = divmod(len(files), no_of_jobs)
    return (files[i * k + min(i, m):(i + 1) * k +
                  min(i + 1, m)] for i in range(no_of_jobs))


def main(args):
    print(args)
    modes = ['train', 'test']
    cut_square = args.square

    for mode in modes:
        print("Running for mode ", mode)
        ann_list = []
        path = os.path.join(args.src_coco, '*_*_'+mode+".json")
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

            no_of_jobs = args.jobs
            images_chunks = split_list(data_all['annotations'], no_of_jobs)
            with Manager() as manager:
                for j in images_chunks:
                    p = Process(target=crop, args=(j, 
                                                   i,
                                                   mode,
                                                   images,
                                                   mapping_category,
                                                   args.src_img,
                                                   args.dst_img,
                                                   args.zoom,
                                                   cut_square))
                    p.start()
                p.join()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
