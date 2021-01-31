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
from utils.dataset_converter import concatenate_datasets
from utils.split_coco_dataset import split_coco_dataset

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Prepare images of trash for detection task')
    parser.add_argument('--dataset_dest',
                        help='paths to annotations',
                        nargs='+',
                        default=['annotations/annotations-epi.json'])
    parser.add_argument('--split_dest',
                        help='path to destination directory',
                        default='annotations/',
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
    
    # split files into train and test files
    # if you want to concat more datasets simply
    # add path to datasets to the list below
    train_to_concat = []
    test_to_concat = []
    
    for i, data_file in enumerate(args.dataset_dest):
        print('Parsing', data_file, 'file', i+1, 'of', len(args.dataset_dest))
        filename = str(i) + '_' + data_file.split('/')[-1].split('.json')[0]
        print(filename)
        train, test = split_coco_dataset([data_file],
                                         args.split_dest + filename,
                                         args.test_split)
        
        train_source = args.split_dest + filename + '_train.json'
        test_source = args.split_dest + filename + '_test.json'
        train_dest = args.split_dest + "binary_" + filename +"_train.json"
        test_dest = args.split_dest + "binary_" + filename +"_test.json"
        
        convert_to_binary(source=train_source,
                        dest=train_dest)
        convert_to_binary(source=test_source,
                        dest=test_dest)
                        
        train_to_concat.append(train_dest)
        test_to_concat.append(test_dest)     
                        
    concatenate_datasets(train_to_concat, dest = args.split_dest + 'binary_mixed_train.json')
    concatenate_datasets(test_to_concat, dest = args.split_dest + 'binary_mixed_test.json')

    
