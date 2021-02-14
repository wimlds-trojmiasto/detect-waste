'''
Script to prepare pseudolabels for litter classification.
'''
import argparse
import json
import sys
import os

import torch.nn.parallel
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import matplotlib.pyplot as plt
from contextlib import suppress
from tqdm import tqdm

from effdet.data.transforms import *
from effdet import (create_model, create_evaluator,
                    create_dataset, create_loader)
from effdet.data import resolve_input_config
from effdet.evaluator import CocoEvaluator, PascalEvaluator
from timm.utils import AverageMeter, setup_default_logging


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Prepare predicted annotation for openlitter map')
    parser.add_argument('--dst_coco', help='path to save coco annotations',
                        type=str, default='/dih4/dih4_home/smajchrowska/openlittermap.json')
    parser.add_argument('--src_img',
                        help='path to source directory with images',
                        type=str,
                        default='/dih4/dih4_home/smajchrowska/test')
    parser.add_argument('--checkpoint',
                        help='path to efficientdet checkpoint',
                        type=str,
                        default='/dih4/dih4_2/wimlds/smajchrowska/output/train/20210130-231654-tf_efficientdet_d2/model_best.pth.tar')
    parser.add_argument('--dst_img',
                        help='path to directory to save images with bboxes',
                        type=str,
                        default=None)
    parser.add_argument('--classes',
                        help='list of classes: detect-waste or litter',
                        default='litter',
                        choices=['detect-waste', 'litter'],
                        type=str)
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    return parser


def main(args):
    print(args)
    if args.classes == 'litter':
        CLASSES = ['Litter']
        categories = [
            {'category': 'Litter',
             'id': 1, 'name': 'Litter', 'supercategory': 'Litter'}]
    elif args.classes == 'detect-waste':
        CLASSES = ['metals_and_plastic', 'other', 'non_recyclable',
                   'glass',  'paper',  'bio', 'unknown']
        categories = [
            {'category': 'metals_and_plastic',
             'id': 1, 'name': 'metals_and_plastic', 'supercategory': 'Litter'},
            {'category': 'other', 'id': 2, 'name': 'other',
             'supercategory': 'Litter'},
            {'category': 'non_recyclable', 'id': 3, 'name': 'non_recyclable',
             'supercategory': 'Litter'},
            {'category': 'glass', 'id': 4, 'name': 'glass',
             'supercategory': 'Litter'},
            {'category': 'paper', 'id': 5, 'name': 'paper',
             'supercategory': 'Litter'},
            {'category': 'bio', 'id': 6, 'name': 'bio',
             'supercategory': 'Litter'},
            {'category': 'unknown', 'id': 7, 'name': 'unknown',
             'supercategory': 'Litter'}]
    else:
        sys.exit('Unknown category list.')
    if args.dst_img is not None and not os.path.exists(args.dst_img):
        os.mkdir(args.dst_img)
    num_classes = len(CLASSES)

    model = args.checkpoint.split('-')[-1].split('/')[0]
    # create model
    bench = create_model(
        model,
        bench_task='predict',
        num_classes=num_classes,
        pretrained=False,
        redundant_bias=True,
        checkpoint_path=args.checkpoint
    )

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (model, param_count))

    bench = bench.to(args.device)

    torch.set_grad_enabled(False)
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize((768, 768)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    annotations = {}
    annotations['info'] = {
        'contributor': 'WiML&DS, Detect Waste in Pomerania',
        'date_created': '12.02.2021',
        'description': 'Detect waste is a non-profit, educational, '
                       'eco project that aims to use Artificial '
                       'Intelligence for the general good. '
                       'We have gathered a team of ten carefully selected '
                       'members and five mentors to work together on the '
                       'problem of the world’s waste pollution.',
        'url': 'detectwaste.ml',
        'version': 'v1',
        'year': '2021'
    }
    annotations['licenses'] = [
        {'id': 0,
         'name': 'Attribution-NonCommercial 4.0 International',
         'url': 'https://creativecommons.org/licenses/by-nc/4.0/legalcode'}]
    annotations['categories'] = categories
    annotations['images'] = []
    img_id = 0
    annotations['annotations'] = []
    ann_id = 0
    for fname in tqdm(os.listdir(args.src_img)):
        img_name = os.path.join(args.src_img, fname)
        # read an image
        try:
            im = Image.open(img_name).convert('RGB')
        except:
            print(f"Error with {img_name}")
            continue
        w, h = im.size
        img_item = {'file_name': fname,
                    'height': h,
                    'id': img_id,
                    'width': w}
        annotations['images'].append(img_item)
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0).to(args.device)
        # propagate through the model
        outputs = bench(img)
        if args.dst_img is not None:
            plt.figure(figsize=(16, 10))
            plt.imshow(im)
            ax = plt.gca()
        for i in outputs[0, outputs[0, :, 4] > args.score_thr].tolist():
            scale_w = w/768
            scale_h = h/768
            i[0] *= scale_w
            i[1] *= scale_h
            i[2] *= scale_w
            i[3] *= scale_h
            ann_item = {'area': (i[2] - i[0]) * (i[3] - i[1]),
                        'bbox': [i[0], i[1], i[2] - i[0], i[3] - i[1]],
                        'category_id': int(i[-1]),
                        'id': ann_id,
                        'image_id': img_id,
                        'iscrowd': 0,
                        'probability': i[4]}
            annotations['annotations'].append(ann_item)
            if args.dst_img is not None:
                p = np.array(i[4:-1])
                ax.add_patch(plt.Rectangle(
                    (i[0], i[1]), i[2] - i[0], i[3] - i[1],
                    fill=False, color='r', linewidth=3))
                cl = int(i[-1])-1
                text = f'{CLASSES[cl]}: {p[0]:0.2f}'
                ax.text(i[0], i[1], text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))
            ann_id += 1
        img_id += 1
        if args.dst_img is not None:
            plt.axis('off')
            plt.savefig(os.path.join(args.dst_img, 'BBox_' + fname))
            plt.close()

    with open(args.dst_coco, 'w') as f:
        json.dump(annotations, f)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
