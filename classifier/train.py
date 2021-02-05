import torch
import numpy as np
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, confusion_matrix
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger
from efficientnet_pytorch import EfficientNet
from torchvision import datasets
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# from neptunecontrib.monitoring.metrics import log_binary_classification_metrics, log_classification_report
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import MultiplicativeLR
import matplotlib.pyplot as plt

TRAIN_DIR = 'images_square/train/'
TEST_DIR = 'images_square/test/'
LITTERMAP_DIR = 'openlittermap/'
BATCH_SIZE = 8
CLASSES = 7
MODEL_NAME = 'efficientnet-b4'
GPUs = [5]
EPOCHS = 20

class LitterClassification(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained(MODEL_NAME,num_classes=CLASSES)

    def forward(self, x):
        x = x['image']
        out = self.efficient_net(x)
        return out
    
    def configure_optimizers(self):        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lambd = lambda epoch: 0.99
        scheduler = MultiplicativeLR(optimizer, lr_lambda = lambd)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x,y =  batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred,y)
        acc = accuracy(y_pred,y)
        acc_weighted = accuracy(y_pred,y, class_reduction = 'weighted')
        self.log("train_acc",acc,prog_bar=True,logger=True)
        self.log("train_acc_weighted",acc_weighted,prog_bar=True,logger=True)
        self.log("train_loss",loss,prog_bar=True,logger=True)

        return {'loss':loss}

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred.to(self.device),y.to(self.device))
        acc = accuracy(y_pred,y)
        acc_weighted = accuracy(y_pred,y, class_reduction = 'weighted')
       
        self.log("val_acc",acc,prog_bar=True,logger=True)
        self.log("val_acc_weighted",acc_weighted,prog_bar=True,logger=True)
        self.log("val_loss",loss,prog_bar=True,logger=True)
        return {'loss':loss, 'y':y, 'y_pred':y_pred}
    
    def validation_epoch_end(self, outputs):
        scores = []
        for i, out in enumerate(outputs):
            y, y_pred = out['y'], out['y_pred']
            
            if i == 0:
                all_y = y
                all_ypred = y_pred
            else:
                all_y = torch.cat((all_y,y),0)
                all_ypred = torch.cat((all_ypred,y_pred),0)
            
            conf_mat = confusion_matrix(y_pred,y,
                                    num_classes = CLASSES)
            # scores += conf_mat
        # from one-hot to labels
        all_ypred = torch.argmax(all_ypred, dim=1).cpu().detach().numpy()
        all_y = all_y.cpu().detach().numpy()
        self.logger.experiment.log_text("classification_report",str(classification_report(all_y,all_ypred)))
        fig, ax = plt.subplots(figsize=(10,10))
        plot_confusion_matrix(all_y, all_ypred, ax=ax)
        self.logger.experiment.log_image('confusion_matrix', fig)
        # print(scores)
          
def get_augmentation(transform):
    return lambda img:transform(image=np.array(img))

if __name__ == '__main__':
    img_size = EfficientNet.get_image_size(MODEL_NAME)
    train_transform = A.Compose([A.Resize(img_size+60,img_size+60),
                            A.RandomCrop(img_size,img_size),
                            A.HorizontalFlip(),
                            A.VerticalFlip(),
                            A.ShiftScaleRotate(),
                            A.RandomBrightnessContrast(),
                            A.Cutout(),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()])
    test_transform = A.Compose([A.Resize(img_size,img_size),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2()])
    train_set = datasets.ImageFolder(root = TRAIN_DIR,
                                    transform = get_augmentation(train_transform))
    test_set = datasets.ImageFolder(root = TEST_DIR,
                                    transform = get_augmentation(test_transform))
    openlittermap = datasets.ImageFolder(root = LITTERMAP_DIR,
                                    transform = get_augmentation(train_transform))
    # concatenate datasets to use openlittermap data for training
    train_set = torch.utils.data.ConcatDataset([train_set,openlittermap])
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE)
    model = LitterClassification()
    model_checkpoint = ModelCheckpoint(monitor = "val_loss",
                                   verbose=True,
                                   filename="{epoch}_{val_loss:.4f}")
    
    # your NEPTUNE_API_TOKEN should be add to ~./bashrc to run this file
    logger = NeptuneLogger(project_name = 'detectwaste/classification',
                           tags=[MODEL_NAME, TRAIN_DIR,'RandomBrightnessContrast', 'openlittermap'])
    
    #CPU:default,GPU:gpus,TPU:tpu_cores
    trainer = pl.Trainer(gpus=GPUs,
                        max_epochs=EPOCHS,
                        callbacks=[model_checkpoint],
                        logger = logger) 
    trainer.fit(model,train_loader, test_loader) 

    #manually you can save best checkpoints - 
    trainer.save_checkpoint("effnet.ckpt")
