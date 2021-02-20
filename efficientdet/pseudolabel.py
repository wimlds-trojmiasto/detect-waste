'''
Script to prepare pseudolabels for litter classification.
'''
import argparse
import json
import os

import torch.nn.parallel
from PIL import Image
import torch
from tqdm import tqdm

from demo import (set_model, get_transforms,
                  rescale_bboxes, plot_results)


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Prepare predicted annotation for unknown dataset')
    parser.add_argument('--dst_coco',
                        help='path to save coco pseudoannotations',
                        type=str)
    parser.add_argument('--src_img',
                        help='path to source directory with images',
                        type=str)
    parser.add_argument('--checkpoint',
                        help='path to efficientdet checkpoint',
                        type=str)
    parser.add_argument('--dst_img',
                        help='path to directory to save images with bboxes',
                        type=str,
                        default=None)
    parser.add_argument('--classes', nargs='+', default=['Litter'])
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    return parser


def main(args):
    print(args)
    CLASSES = args.classes
    categories = []
    iid = 0
    for i in args.classes:
        iid += 1
        item = {'category': i,
                'id': iid, 'name': i, 'supercategory': 'Litter'}
        categories.append(item)

    if args.dst_img is not None and not os.path.exists(args.dst_img):
        os.mkdir(args.dst_img)
    num_classes = len(CLASSES)
    torch.set_grad_enabled(False)
    model = args.checkpoint.split('-')[-1].split('/')[0]
    bench = set_model(model, num_classes, args.checkpoint, args.device)

    bench.eval()

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
        # read an image and add it to annotations
        try:
            im = Image.open(img_name).convert('RGB')
            w, h = im.size
            img_item = {'file_name': fname,
                        'height': h,
                        'id': img_id,
                        'width': w}
            annotations['images'].append(img_item)
        except:
            print(f"Error with {img_name}")
            continue

        # mean-std normalize the input image (batch-size: 1)
        img = get_transforms(im)

        # propagate through the model
        outputs = bench(img.to(args.device))

        # keep only predictions above set confidence
        bboxes_keep = outputs[0, outputs[0, :, 4] > args.prob_threshold]
        probas = bboxes_keep[:, 4:]

        # convert boxes to image scales
        bboxes_scaled = rescale_bboxes(bboxes_keep[:, :4], im.size,
                                       tuple(img.size()[2:]))

        # plot and save demo image
        if args.dst_img is not None:
            try:
                plot_results(im, probas, bboxes_scaled, args.classes,
                             os.path.join(args.dst_img, 'BBox_' + fname))
            except:
                print(f"Error with {img_name}")
                continue

        for p, i in zip(probas, bboxes_scaled):
            ann_item = {'area': (i[2] - i[0]) * (i[3] - i[1]),
                        'bbox': [i[0], i[1], i[2] - i[0], i[3] - i[1]],
                        'category_id': int(p[1]),
                        'id': ann_id,
                        'image_id': img_id,
                        'iscrowd': 0,
                        'probability': format(float(p[0]), '.2f')}
            annotations['annotations'].append(ann_item)
            ann_id += 1
        img_id += 1

        with open(args.dst_coco, 'w') as f:
            json.dump(annotations, f)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
