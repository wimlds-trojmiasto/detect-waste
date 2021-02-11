import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger
from efficientnet_pytorch import EfficientNet
from torchvision import datasets
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from models.efficientnet import LitterClassification

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Train efficientnet')
    
    # Dataset / Model parameters
    parser.add_argument('--data', metavar='DIR',
                        help='path to base directory with data',
                        default='/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/classifier/')
    parser.add_argument('--model', default='efficientnet-b0', type=str, 
                        help='Name of model to train (default: "efficientnet-b0)"')
    parser.add_argument('--lr', type=float, default=0.0001, 
                    help='learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=0.99, 
                    help='learning rate (default: 0.99)')
    parser.add_argument('-b', '--batch-size', type=int, default=16, 
                    help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
    parser.add_argument('--num-classes', type=int, default=7, metavar='N',
                    help='number of classes to classify (default: 7)')
    parser.add_argument('--gpu', type=int, default=7, metavar='N',
                    help='GPU number to use (default: 7)')
    parser.set_defaults(redundant_bias=None)
    return parser
      
def get_augmentation(transform):
    return lambda img:transform(image=np.array(img))

def main(args):
    TRAIN_DIR = os.path.join(args.data, 'images_square','train')
    TEST_DIR =os.path.join(args.data, 'images_square','test')
    PSEUDO_DIR = os.path.join(args.data, 'images_square','pseudolabel')
    img_size = EfficientNet.get_image_size(args.model)
    train_transform = A.Compose([A.Resize(img_size+60,img_size+60),
                            A.RandomCrop(img_size,img_size),
                            A.HorizontalFlip(),
                            A.VerticalFlip(),
                            A.ShiftScaleRotate(),
                            A.RandomBrightnessContrast(),
                            A.Cutout(),
                            A.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                            ToTensorV2()])
    test_transform = A.Compose([A.Resize(img_size,img_size),
                            A.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                            ToTensorV2()])
    train_set = datasets.ImageFolder(root = TRAIN_DIR,
                                    transform = get_augmentation(train_transform))
    test_set = datasets.ImageFolder(root = TEST_DIR,
                                    transform = get_augmentation(test_transform))
    
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=args.batch_size)
    test_loader = DataLoader(test_set, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             num_workers=args.batch_size)

    if PSEUDO_DIR != None:
        pseudo_set = datasets.ImageFolder(root = PSEUDO_DIR,
                                    transform = get_augmentation(train_transform))
        pseudo_loader = DataLoader(pseudo_set,
                                   batch_size=args.batch_size, 
                                   shuffle=True, 
                                   num_workers=args.batch_size)
        
        model = LitterClassification(model_name=args.model, 
                                     lr=args.lr,
                                     decay=args.decay,
                                     num_classes=args.num_classes,
                                     pseudoloader=pseudo_loader)
    else:
        model = LitterClassification(model_name=args.model,
                                     lr=args.lr,
                                     decay=args.decay,
                                     num_classes=args.num_classes)
        
    model_checkpoint = ModelCheckpoint(monitor = "val_loss",
                                   verbose=True,
                                   filename="{epoch}_{val_loss:.4f}")
    
    # your NEPTUNE_API_TOKEN should be add to ~./bashrc to run this file
    logger = NeptuneLogger(project_name = 'detectwaste/classification',
                           tags=[args.model, TRAIN_DIR])
    
    #CPU:default,GPU:gpus,TPU:tpu_cores
    trainer = pl.Trainer(gpus=[args.gpu],
                        max_epochs=args.epochs,
                        callbacks=[model_checkpoint],
                        logger = logger) 
    trainer.fit(model,train_loader, test_loader) 

    #manually you can save best checkpoints - 
    trainer.save_checkpoint("effnet.ckpt")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)