'''
Script to prepare annotation for litter detection task.
'''
import argparse
import os
import numpy as np

# update all annotations in one run
from utils.dataset_converter import convert_dataset, \
                                    taco_categories_to_detectwaste, \
                                    convert_to_binary
from utils.split_coco_dataset import split_coco_dataset


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Prepare images of trash for classification task')
    parser.add_argument('--epi_source', help='path to epinote annotations',
                        default='/dih4/dih4_2/wimlds/data'
                                '/annotations_epi.json',
                        type=str)
    parser.add_argument('--taco_source',
                        help='path to taco annotations',
                        default='/dih4/dih4_2/wimlds/TACO-master'
                                '/data/annotations.json',
                        type=str)
    parser.add_argument('--detectwaste_dest',
                        help='path to detectwaste annotations',
                        default='annotations/annotations_detectwaste.json',
                        type=str)
    parser.add_argument('--epi_dest',
                        help='path to source epi annotations',
                        default='annotations/annotations-epi.json',
                        type=str)
    parser.add_argument('--split_dest',
                        help='path to destination directory',
                        default='annotations/annotations',
                        type=str)
    parser.add_argument('--test_split',
                        help='fraction of dataset for test',
                        default=0.2,
                        type=str)
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    np.random.seed(2020)

    # create directory to store all annotations
    if not os.path.exists(os.path.dirname(args.detectwaste_dest)):
        os.mkdir(os.path.dirname(args.detectwaste_dest))

    # first, move all category ids from taco annotation style (60 categories)
    # to detectwaste (7 categories)
    taco_categories_to_detectwaste(source=args.taco_source,
                                   dest=args.detectwaste_dest)
    # convert from epi to detectwaste
    convert_dataset(args.detectwaste_dest, args.epi_source, args.epi_dest)

    # split files into train and test files
    # if you want to concat more datasets simply
    # add path to datasets to the list below
    list_of_datasets = [args.detectwaste_dest,
                        args.epi_dest,
                        ]

    split_coco_dataset(list_of_datasets,
                       args.split_dest,
                       args.test_split)

    # convert all annotations to binary to preserve original split
    convert_to_binary(source=args.split_dest+'_train.json',
                      dest=args.split_dest+'_binary_train.json')
    convert_to_binary(source=args.split_dest+'_test.json',
                      dest=args.split_dest+'_binary_test.json')
