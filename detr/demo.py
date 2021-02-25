'''
Demo based on DETR's hands on Colab Notebook:
https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb
'''
import argparse
import cv2
import os
import time

from PIL import Image
import PIL.ImageColor as ImageColor
import requests
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Test detr on one image')
    parser.add_argument(
        '--img', metavar='IMG',
        help='path to image, could be url',
        default='https://www.fyidenmark.com/images/denmark-litter.jpg')
    parser.add_argument(
        '--save', metavar='OUTPUT',
        help='path to save image with predictions (if None show image)',
        default=None)
    parser.add_argument(
        '--model', default='detr_resnet50', type=str,
        help='Name of model to test (default: "detr_resnet50")')
    parser.add_argument('--classes', nargs='+', default=['Litter'])
    parser.add_argument(
        '--checkpoint', type=str,
        help='path to checkpoint')
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='device to evaluate model (default: cpu)')
    parser.add_argument(
        '--prob_threshold', type=float, default=0.5,
        help='probability threshold to show results (default: 0.5)')
    parser.add_argument(
        '--video', action='store_true', default=False,
        help="If true, we treat impute as video (default: False)")
    parser.set_defaults(redundant_bias=None)
    return parser


# standard PyTorch mean-std input image normalization
def get_transforms(im):
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(im).unsqueeze(0)


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h],
                         dtype=torch.float32).to(b.device)
    return b


# from https://deepdrive.pl/
def get_output(img, prob, boxes, classes=['Litter'], stat_text=None):
    # colors for visualization
    STANDARD_COLORS = [
        'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige',
        'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
        'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk',
        'Crimson', 'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey',
        'DarkKhaki', 'DarkOrange', 'DarkOrchid', 'DarkSalmon',
        'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold',
        'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink',
        'IndianRed', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush',
        'LawnGreen', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray',
        'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen',
        'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue',
        'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta',
        'MediumAquaMarine', 'MediumOrchid', 'MediumPurple',
        'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream',
        'MistyRose', 'Moccasin', 'NavajoWhite', 'OldLace', 'Olive',
        'OliveDrab', 'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenRod',
        'PaleGreen', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip',
        'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue',
        'GreenYellow', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet',
        'Wheat', 'White', 'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]
    palette = [ImageColor.getrgb(_) for _ in STANDARD_COLORS]
    for p, (x0, y0, x1, y1) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        color = palette[cl]
        start_p, end_p = (int(x0), int(y0)), (int(x1), int(y1))
        cv2.rectangle(img, start_p, end_p, color, 2)
        text = "%s %.1f%%" % (classes[cl], p[cl]*100)
        cv2.putText(img, text, start_p, cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 10)
        cv2.putText(img, text, start_p, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    if stat_text is not None:
        cv2.putText(img, stat_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 10)
        cv2.putText(img, stat_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 3)
    return img


# from https://deepdrive.pl/
def save_frames(args, num_iter=45913):
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    cap = cv2.VideoCapture(args.img)
    counter = 0
    pbar = tqdm(total=num_iter+1)
    num_classes = len(args.classes)
    model = set_model(args.model, num_classes, args.checkpoint, args.device)
    model.eval()

    model.to(args.device)

    while(cap.isOpened()):
        ret, img = cap.read()
        if img is None:
            print("END")
            break

        # scale + BGR to RGB
        inference_size = (800, 800)
        scaled_img = cv2.resize(img[:, :, ::-1], inference_size)

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # mean-std normalize the input image (batch-size: 1)
        img_tens = transform(scaled_img).unsqueeze(0).to(args.device)

        # Inference
        t0 = time.time()
        with torch.no_grad():
            # propagate through the model
            output = model(img_tens)
        t1 = time.time()

        probas = output['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > args.prob_threshold
        bboxes_scaled = rescale_bboxes(output['pred_boxes'][0, keep],
                                       (img.shape[1], img.shape[0]))
        txt = "Detect-waste %s Threshold=%.2f " \
              "Inference %dx%d  GPU: %s Inference time %.3fs" % \
              (args.model, args.prob_threshold, inference_size[0],
               inference_size[1], torch.cuda.get_device_name(0),
               t1 - t0)
        result = get_output(img, probas[keep], bboxes_scaled,
                            args.classes, txt)
        cv2.imwrite(os.path.join(args.save, 'img%08d.jpg' % counter), result)
        counter += 1
        pbar.update(1)
        del img
        del img_tens
        del result

    cap.release()


def plot_results(pil_img, prob, boxes, classes=['Litter'],
                 save_path=None, colors=None):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    if colors is None:
        # colors for visualization
        colors = 100 * [
           [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
           [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{classes[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight',
                    transparent=True, pad_inches=0)
        plt.close()
        print(f'Image saved at {save_path}')
    else:
        plt.show()


def set_model(model_type, num_classes, checkpoint_path, device):
    model = torch.hub.load('facebookresearch/detr',
                           model_type,
                           pretrained=False,
                           num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path,
                            map_location=device)

    model.load_state_dict(checkpoint['model'],
                          strict=False)
    return model


def main(args):
    # prepare model for evaluation
    torch.set_grad_enabled(False)
    num_classes = len(args.classes)
    model = set_model(args.model, num_classes, args.checkpoint, args.device)
    model.eval()

    model.to(args.device)
    # get image
    if args.img.startswith('https'):
        im = Image.open(requests.get(args.img, stream=True).raw).convert('RGB')
    else:
        im = Image.open(args.img).convert('RGB')

    # mean-std normalize the input image (batch-size: 1)
    img = get_transforms(im)

    # propagate through the model
    outputs = model(img.to(args.device))

    # keep only predictions above set confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > args.prob_threshold

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # plot and save demo image
    plot_results(im, probas[keep], bboxes_scaled, args.classes, args.save)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.video:
        save_frames(args)
    else:
        main(args)
