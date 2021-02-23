import os
import argparse
import numpy as np
import torch
from torch import DoubleTensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import (SubsetRandomSampler,
                                      WeightedRandomSampler)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger
from efficientnet_pytorch import EfficientNet
from torchvision import datasets
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import json
from models.efficientnet import LitterClassification
from train_resnet import make_weights_for_balanced_classes
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Train efficientnet')

    # Dataset / Model parameters
    parser.add_argument(
        '--data', metavar='DIR',
        help='path to base directory with data',
        default='/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/classifier/')
    parser.add_argument(
        '--checkpoint', metavar='OUTPUT',
        help='path to models checkpoint',
        default='/dih4/dih4_2/wimlds/smajchrowska/classifier/background_b2_weighted_per_batch.ckpt')
    parser.add_argument(
        '--model', default='efficientnet-b0', type=str,
        help='Name of model to train (default: "efficientnet-b0)"')
    parser.add_argument(
        '-b', '--batch-size', type=int, default=16,
        help='input batch size for training (default: 16)')
    parser.add_argument(
        '--num-classes', type=int, default=7, metavar='NUM',
        help='number of classes to classify (default: 7)')
    parser.add_argument(
        '--gpu', type=int, default=7, metavar='GPU',
        help='GPU number to use (default: 7)')
    parser.set_defaults(redundant_bias=None)
    return parser


def make_sampler(split_set, weighted_sampler=False):
    indices = list(range(len(split_set)))
    if weighted_sampler:
        # For unbalanced dataset we create a weighted sampler
        sampler = []
        for i in indices:
            sampler.append(split_set.imgs[i])
        weights = make_weights_for_balanced_classes(sampler,
                                                    len(split_set.classes))
        sampler = WeightedRandomSampler(DoubleTensor(weights), len(weights))
    else:
        sampler = SubsetRandomSampler(indices)
    return sampler


def get_augmentation(transform):
    return lambda img: transform(image=np.array(img))


def main(args):
    IM_DIR = os.path.join(args.data, 'images_square', 'train')
    img_size = EfficientNet.get_image_size(args.model)
    transform = A.Compose([A.Resize(img_size, img_size),
                                A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                                ToTensorV2()])

    image_set = datasets.ImageFolder(
        root=IM_DIR,
        transform=get_augmentation(transform))

    image_loader = DataLoader(image_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.batch_size)

    model = LitterClassification.load_from_checkpoint(args.checkpoint,
                                                    model_name=args.model, lr = 0, decay = 0)
    model.eval()
    idx_to_class = {0: 'background', 
                    1: 'bio',
                    2: 'glass',
                    3: 'metals_and_plastic',
                    4: 'non_recyclable',
                    5: 'other',
                    6: 'paper',
                    7: 'unknown'}
    results = {}
    for i, batch in tqdm(enumerate(image_loader)):
        image, folder_num = batch
        predictions = model(image)
        predictions = torch.argmax(predictions, dim=1).cpu().detach().numpy()
        start = i*args.batch_size
        end = (i+1)*args.batch_size
        image_paths = image_loader.dataset.imgs[start:end]
        image_paths = [im[0] for im in image_paths]
        
        for j, idx in enumerate(range(start,end)):
            batch_results = {}
            try:
                batch_results['img'] = image_paths[j]
                batch_results['prediction'] = idx_to_class[predictions[j]]
                results[idx] = batch_results
            except:
                continue
            
    with open(args.model+'_predictions.json', 'w', encoding='utf8') as json_file:
        json.dump(results, json_file, indent=6)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
