import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, confusion_matrix
from scikitplot.metrics import plot_confusion_matrix
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import MultiplicativeLR
import matplotlib.pyplot as plt
from tqdm import tqdm

class LitterClassification(pl.LightningModule):
    
    def __init__(self, model_name, lr, decay, num_classes = 7, pseudoloader = None, pseudolabelling_start = 5):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained(model_name,num_classes=num_classes)
        self.pseudoloader = pseudoloader
        self.pseudolabelling_start = pseudolabelling_start
        self.lr = lr
        
    def forward(self, x):
        x = x['image'].to(self.device)
        out = self.efficient_net(x)
        return out
    
    def configure_optimizers(self):        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lambd = lambda epoch: self.decay
        scheduler = MultiplicativeLR(optimizer, lr_lambda = lambd)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx, pseudo_label = False):
        x,y =  batch
        y_pred = self(x)
        y_pred, y = y_pred.to(self.device), y.to(self.device)
        loss = F.cross_entropy(y_pred,y)
        acc = accuracy(y_pred,y)
        acc_weighted = accuracy(y_pred,y, class_reduction = 'weighted')
        
        if pseudo_label:            
            self.log("pseudo_loss",loss,prog_bar=True,logger=True)
        else:
            self.log("train_acc",acc,prog_bar=True,logger=True)
            self.log("train_acc_weighted",acc_weighted,prog_bar=True,logger=True)
            self.log("train_loss",loss,prog_bar=True,logger=True)

        return {'loss':loss, 'y':y, 'y_pred':y_pred}

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
            
        # from one-hot to labels
        all_ypred = torch.argmax(all_ypred, dim=1).cpu().detach().numpy()
        all_y = all_y.cpu().detach().numpy()
        
        # plot confusion matrix and log to neptune
        fig, ax = plt.subplots(figsize=(10,10))
        plot_confusion_matrix(all_y, all_ypred, ax=ax)
        self.logger.experiment.log_image('confusion_matrix', fig)
        self.logger.experiment.log_text("classification_report",str(classification_report(all_y,all_ypred)))
        
        if self.current_epoch >= self.pseudolabelling_start:
            self.pseudolabelling_update_outputs()
            self.pseudolabelling_update_loss()
        
    def pseudolabelling_update_loss(self):
        print('Calculating loss for pseudo-labelling')
        optimizer = self.optimizers()
        for i, batch in tqdm(enumerate(self.pseudoloader)):
            output = self.training_step(batch, i, pseudo_label = True)
            loss = output['loss']
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.log("pseudo_loss",loss,
                    prog_bar=True,logger=True)
            
    def pseudolabelling_update_outputs(self):
        print('Updating outputs for pseudolabelling')
        optimizer = self.optimizers()
        for batch_idx, batch in tqdm(enumerate(self.pseudoloader)):
            output = self.training_step(batch, batch_idx, pseudo_label = True)
            y_pred = output['y_pred']
            # apply new targets
            for idx, y in enumerate(y_pred):
                self.pseudoloader.dataset.targets[batch_idx*len(y_pred)+idx] = torch.argmax(y, dim=0)
                           
        # now switch class_to_idx and classes in pseudo-dataset
        # to the same as in traindataset   
        self.pseudoloader.dataset.class_to_idx = {'bio': 0,
                                              'glass': 1,
                                              'metals_and_plastic': 2, 
                                              'non_recyclable': 3, 
                                              'other': 4, 
                                              'paper': 5, 
                                              'unknown': 6}
        self.pseudoloader.dataset.classes = ['bio', 'glass',
                                            'metals_and_plastic', 
                                            'non_recyclable', 'other',
                                            'paper', 'unknown']