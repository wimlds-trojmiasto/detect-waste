from engine import evaluate, train_one_epoch
import argparse
import utils
from data import build
import datetime
from pathlib import Path
import time
import os
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import neptune


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Prepare instance segmentation task with Mask R-CNN')
    parser.add_argument('--output_dir',
                        help='path to save checkpoints',
                        default='/dih4/dih4_2/wimlds/smajchrowska/',
                        type=str)
    parser.add_argument('--images_dir',
                        help='path to images directory',
                        default='/dih4/dih4_2/wimlds/data/all_detect_images',
                        type=str)
    parser.add_argument(
        '--anno_name',
        help='path to annotation json (part name)',
        default='/dih4/dih4_home/smajchrowska/detect-waste/'
                'annotations/annotations_binary_mask_all',
        type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--gpu_id', default=2, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--neptune', action='store_true')

    return parser


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.neptune:
        # your NEPTUNE_API_TOKEN should be add to ~./bashrc to run this file
        neptune.init(project_qualified_name='detectwaste/maskrcnn')
        neptune.create_experiment(name='maskrcnn_resnet50_fpn')
    else:
        neptune = None

    output_dir = Path(args.output_dir)

    # use our dataset and defined transformations
    dataset_train = build('train', args.images_dir, args.anno_name)
    dataset_val = build('val', args.images_dir, args.anno_name)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn)

    # define model
    device = torch.device(
        f'cuda:{args.gpu_id}'
        ) if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and waste
    num_classes = args.num_classes

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=10,
                        neptune=neptune)
        # update the learning rate
        lr_scheduler.step()
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                   }, os.path.join(output_dir, f'checkpoint{epoch:04}.pth'))
        # evaluate on the test dataset
        coco_evaluator = evaluate(model, data_loader_test, device=device)
        if neptune is not None:
            neptune.log_metric(
                'valid/bbox-mAP@0.5:0.95',
                coco_evaluator.coco_eval['bbox'].stats[0])
            neptune.log_metric(
                'valid/bbox-mAP@0.5',
                coco_evaluator.coco_eval['bbox'].stats[1])
            neptune.log_metric(
                'valid/segm-mAP@0.5:0.95',
                coco_evaluator.coco_eval['segm'].stats[0])
            neptune.log_metric(
                'valid/segm-mAP@0.5',
                coco_evaluator.coco_eval['segm'].stats[1])
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
