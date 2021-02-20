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
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from efficientnet_pytorch import EfficientNet


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
        default='../annotations/annotations_binary_mask0_all',
        type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    # Devices
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    # Learning
    parser.add_argument('--num_epochs', default=26, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument(
        '--lr-step-size', default=0, type=int,
        help='decrease lr every step-size epochs')
    parser.add_argument(
        '--lr-steps', default=[16, 22], nargs='+', type=int,
        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument(
        '--optimizer',
        help='Chose type of optimization algorithm, SGD as default',
        default='SGD', choices=['AdamW', 'SGD'], type=str)
    # Model
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', type=str)
    # Launch neptune
    parser.add_argument('--neptune', action='store_true')

    return parser


# from https://github.com/lukemelas/EfficientNet-PyTorch
def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(
        min_depth,
        int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def get_instance_segmentation_model(
        num_classes, model_name='maskrcnn_resnet50_fpn'):
    # load a pre-trained model for classification
    # and return only the features
    if model_name.startswith('efficientnet'):
        backbone = EfficientNet.from_pretrained(
                model_name, num_classes=num_classes,
                include_top=False)
        # number of output channels
        backbone.out_channels = int(round_filters(1280,
                                                  backbone._global_params))
        model = MaskRCNN(backbone, num_classes)
    else:
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.__dict__[model_name](
            pretrained=True)

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
    start_epoch = 0
    if args.neptune and (not args.resume):
        import neptune
        # your NEPTUNE_API_TOKEN should be add to ~./bashrc to run this file
        neptune.init(project_qualified_name='detectwaste/maskrcnn')
        neptune.create_experiment(name=f'{args.model}')
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
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    # define model
    device = torch.device(
        f'cuda:{args.gpu_id}'
        ) if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and waste
    num_classes = args.num_classes

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes, args.model)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.lr,
                                      weight_decay=args.weight_decay)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)

    # and a learning rate scheduler
    if args.lr_step_size != 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        # evaluate on the test dataset
        print("Start evaluating")
        dataset_val = build('test', args.images_dir, args.anno_name)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            collate_fn=utils.collate_fn)
        evaluate(model, data_loader_test, device=device)
    else:
        print("Start training")
        start_time = time.time()
        for epoch in range(start_epoch, args.num_epochs):
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
