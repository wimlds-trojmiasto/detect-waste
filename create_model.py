import argparse
import sys
sys.path.append('./efficientdet')
sys.path.append('./classifier')

from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet import create_model
from effdet.efficientdet import HeadNet
from models.efficientnet import LitterClassification


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Test modified efficientdet on one image')
    parser.add_argument(
        '--img', metavar='IMG',
        help='path to image, could be url',
        default='https://www.fyidenmark.com/images/denmark-litter.jpg')
    parser.add_argument(
        '--save', metavar='OUTPUT',
        help='path to save image with predictions (if None show image)',
        default=None)
    parser.add_argument('--classes', nargs='+', default=[
        'bio', 'glass', 'metals and plastic',
        'non recyclable', 'other', 'paper', 'unknown'])
    parser.add_argument(
        '--cls_name', type=str, default='efficientnet-b2',
        help='classifier name (default: efficientnet-b2)')
    parser.add_argument(
        '--det_name', type=str, default='tf_efficientdet_d2',
        help='detector name (default: tf_efficientdet_d2)')
    parser.add_argument(
        '--classifier', type=str,
        help='path to classifier checkpoint')
    parser.add_argument(
        '--detector', type=str,
        help='path to detector checkpoint')
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='device to evaluate model (default: cpu)')
    parser.add_argument(
        '--prob_threshold', type=float, default=0.3,
        help='probability threshold to show results (default: 0.5)')
    parser.add_argument(
        '--video', action='store_true', default=False,
        help="If true, we treat impute as video (default: False)")
    parser.set_defaults(redundant_bias=None)
    return parser


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
        if p[1] != 0:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=c, linewidth=3))
            cl = int(p[1]-1)
            text = f'{classes[cl]}: {p[0]:0.2f}'
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


# standard PyTorch mean-std input image normalization
def get_transforms(im):
    transform = T.Compose([
        T.Resize((768, 768)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(im).unsqueeze(0)


def rescale_bboxes(out_bbox, size, resize):
    img_w, img_h = size
    out_w, out_h = resize
    b = out_bbox * torch.tensor([img_w/out_w, img_h/out_h,
                                 img_w/out_w, img_h/out_h],
                                dtype=torch.float32).to(
                                    out_bbox.device)
    return b


def set_model(model_type, num_classes, checkpoint_path, device):

    # create model
    model = create_model(
        model_type,
        bench_task='predict',
        num_classes=num_classes,
        pretrained=False,
        redundant_bias=True,
        checkpoint_path=checkpoint_path
    )

    param_count = sum([m.numel() for m in model.parameters()])
    print('Model %s created, param count: %d' % (model_type, param_count))
    model = model.to(device)
    return model


def weights_update(model, checkpoint, device):
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    return model


def get_classifier(num_classes,
                   cls_name, checkpoint, device):
    model = EfficientNet.from_name(
        cls_name, num_classes=num_classes+1)
    weights_update(model, torch.load(checkpoint), device)
    return model


def get_modified_efficientdet(num_classes,
                              cls_name, det_name,
                              classifier_checkpoint,
                              detector_checkpoint, device):
    net = set_model(det_name, 1,
                    detector_checkpoint, device)
    breakpoint()
    net.class_net = get_classifier(
        num_classes+1, cls_name, classifier_checkpoint, device)
    return net


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.video:
        # prepare model for evaluation
        torch.set_grad_enabled(False)
        num_classes = len(args.classes)
        model = get_classifier(
            num_classes, args.cls_name, args.classifier, args.device)
        model.eval()
        # get image
        if args.img.startswith('https'):
            import requests
            im = Image.open(requests.get(args.img, stream=True).raw).convert('RGB')
        else:
            im = Image.open(args.img).convert('RGB')

        # mean-std normalize the input image (batch-size: 1)
        img = get_transforms(im)

        # propagate through the model
        outputs = model(img)
        preds = torch.topk(outputs, k=num_classes).indices.squeeze(0).tolist()

        print('-----')
        for idx in preds:
            if idx != 0:
                label = args.classes[idx-1]
            else:
                label = 'background'
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            print('{:<75} ({:.2f}%)'.format(label, prob*100))

        print("TBA")
    else:
        # get full image
        if args.img.startswith('https'):
            import requests
            im = Image.open(requests.get(args.img, stream=True).raw).convert('RGB')
        else:
            im = Image.open(args.img).convert('RGB')
        # prepare model for evaluation
        torch.set_grad_enabled(False)
        # 1) Localize
        detector = set_model(args.det_name, 1,
                             args.detector, args.device)
        detector.eval()
        # mean-std normalize the input image (batch-size: 1)
        img = get_transforms(im)
        # propagate through the model
        outputs = detector(img.to(args.device))

        # keep only predictions above set confidence
        bboxes_keep = outputs[0, outputs[0, :, 4] > args.prob_threshold]
        probas = bboxes_keep[:, 4:]
        # convert boxes to image scales
        bboxes_scaled = rescale_bboxes(bboxes_keep[:, :4], im.size,
                                       tuple(img.size()[2:]))

        # 2) Classify
        num_classes = len(args.classes)
        classifier = get_classifier(
            num_classes+1, args.cls_name, args.classifier, args.device)
        classifier.eval()
        for p, (xmin, ymin, xmax, ymax) in zip(
                            probas, bboxes_scaled.tolist()):
            # mean-std normalize the input image (batch-size: 1)
            img = get_transforms(im.crop((xmin, ymin, xmax, ymax)))

            # propagate through the model
            outputs = classifier(img)
            p[1] = torch.topk(outputs, k=1).indices.squeeze(0).tolist()[0]

        # plot and save demo image
        plot_results(im, probas, bboxes_scaled, args.classes, args.save)
