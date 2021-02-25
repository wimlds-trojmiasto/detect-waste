import argparse
import os
import sys
import time
sys.path.append('./efficientdet')
sys.path.append('./classifier')

import cv2
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm

from demo import (get_output, set_model, rescale_bboxes,
                  plot_results, get_transforms)
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
        '--prob_threshold', type=float, default=0.17,
        help='probability threshold to show results (default: 0.17)')
    parser.add_argument(
        '--cls_th', type=float, default=0.5,
        help='probability threshold to show results (default: 0.5)')
    parser.add_argument(
        '--video', action='store_true', default=False,
        help="If true, we treat impute as video (default: False)")
    parser.set_defaults(redundant_bias=None)
    return parser


def get_classifier(num_classes,
                   cls_name, checkpoint, device):
    model = LitterClassification.load_from_checkpoint(
        checkpoint, model_name=cls_name, lr=0, decay=0)
    return model.to(device)


def save_frames(args, img_size, num_iter=45913):
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    cap = cv2.VideoCapture(args.img)
    counter = 0
    pbar = tqdm(total=num_iter+1)
    num_classes = len(args.classes)
    # detector
    model = set_model(args.det_name, 1, args.detector, args.device)
    model.eval()
    model.to(args.device)
    # classifier
    classifier = get_classifier(
        num_classes+1, args.cls_name, args.classifier, args.device)
    classifier.eval()

    while(cap.isOpened()):
        ret, img_real = cap.read()
        if img_real is None:
            print("END")
            break

        # scale + BGR to RGB
        inference_size = (768, 768)
        scaled_img = cv2.resize(img_real[:, :, ::-1], inference_size)

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

        # keep only predictions above set confidence
        bboxes_keep = output[0, output[0, :, 4] > args.prob_threshold]
        probas = bboxes_keep[:, 4:]

        # convert boxes to image scales
        bboxes_scaled = rescale_bboxes(bboxes_keep[:, :4],
                                       (img_real.shape[1], img_real.shape[0]),
                                       inference_size)
        # 2) Classify
        bboxes_final = []
        cls_prob = []
        img_pill = Image. fromarray(img_real)
        for p, (xmin, ymin, xmax, ymax) in zip(
                            probas, bboxes_scaled.tolist()):

            img = get_transforms(
                img_pill.crop((xmin, ymin, xmax, ymax)), img_size)

            # propagate through the model
            outputs = classifier({'image': img})
            p[1] = torch.topk(outputs, k=1).indices.squeeze(0).tolist()[0]
            p[0] = torch.softmax(outputs, dim=1)[0, int(p[1])].item()
            if p[0] >= args.cls_th:
                bboxes_final.append((xmin, ymin, xmax, ymax))
                cls_prob.append(p)

        txt = "Detect-waste %s Threshold=%.2f " \
              "Inference %dx%d  GPU: %s Inference time %.3fs" % \
              (args.det_name, args.prob_threshold, inference_size[0],
               inference_size[1], torch.cuda.get_device_name(0),
               t1 - t0)
        result = get_output(img_real, probas, bboxes_scaled,
                            args.classes, txt)
        cv2.imwrite(os.path.join(args.save, 'img%08d.jpg' % counter), result)
        counter += 1
        pbar.update(1)
        del img_real
        del img_pill
        del img_tens
        del result

    cap.release()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    img_size = EfficientNet.get_image_size(args.cls_name)
    if args.video:
        save_frames(args, img_size=img_size)
    else:
        # get full image
        if args.img.startswith('https'):
            import requests
            im = Image.open(
                requests.get(args.img, stream=True).raw).convert('RGB')
            dir_list = range(1)
        elif os.path.isdir(args.img):
            dir_list = os.listdir(args.img)
        else:
            im = Image.open(args.img).convert('RGB')
            dir_list = range(1)
        save_path = args.save
        # prepare models for evaluation
        torch.set_grad_enabled(False)
        # detector
        detector = set_model(args.det_name, 1,
                                args.detector, args.device)
        detector.eval()
        # classifier
        num_classes = len(args.classes)
        classifier = get_classifier(
            num_classes+1, args.cls_name, args.classifier, args.device)
        classifier.eval()
        for f in dir_list:
            if os.path.isdir(args.img):
                ifile = os.path.join(args.img, f)
                im = Image.open(ifile).convert('RGB')
                save_path = os.path.join(args.save, f)
            # 1) Localize
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
            bboxes_final = []
            cls_prob = []
            for p, (xmin, ymin, xmax, ymax) in zip(
                                probas, bboxes_scaled.tolist()):

                img = get_transforms(
                    im.crop((xmin, ymin, xmax, ymax)), img_size)

                # propagate through the model
                outputs = classifier({'image': img})
                p[1] = torch.topk(outputs, k=1).indices.squeeze(0).tolist()[0]
                p[0] = torch.softmax(outputs, dim=1)[0, int(p[1])].item()
                if p[0] >= args.cls_th:
                    bboxes_final.append((xmin, ymin, xmax, ymax))
                    cls_prob.append(p)

            # plot and save demo image
            plot_results(im, cls_prob, bboxes_final, args.classes, save_path)
