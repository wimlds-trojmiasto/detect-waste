'''
Demo based on DETR's hands on Colab Notebook:
https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb
'''
import argparse

from PIL import Image
import requests
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T


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
        plt.savefig(save_path)
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
        im = Image.open(requests.get(args.img, stream=True).raw)
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

    main(args)
