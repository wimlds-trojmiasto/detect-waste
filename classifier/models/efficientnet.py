import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers.neptune import NeptuneLogger
from scikitplot.metrics import plot_confusion_matrix
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import MultiplicativeLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


class LitterClassification(pl.LightningModule):
    def __init__(self, model_name, lr, decay, num_classes=8, pseudoloader=None,
                 pseudolabelling_start=5, pseudolabel_mode='per_batch'):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained(
            model_name,
            num_classes=num_classes)
        self.pseudoloader = pseudoloader
        self.pseudolabelling_start = pseudolabelling_start
        self.lr = lr
        self.decay = decay
        self.pseudolabel_mode = pseudolabel_mode

    def forward(self, x):
        x = x['image'].to(self.device)
        out = self.efficient_net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = MultiplicativeLR(optimizer,
                                     lr_lambda=lambda epoch: self.decay)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx, pseudo_label=False):
        x, y = batch
        y_pred = self(x)
        y_pred, y = y_pred.to(self.device), y.to(self.device)
        loss = F.cross_entropy(y_pred, y)
        acc = accuracy(y_pred, y)
        acc_weighted = accuracy(y_pred, y, class_reduction='weighted')

        if pseudo_label:
            self.log("pseudo_loss", loss, prog_bar=True, logger=True)
        else:
            self.log("train_acc", acc, prog_bar=True, logger=True)
            self.log("train_acc_weighted", acc_weighted, prog_bar=True,
                     logger=True)
            self.log("train_loss", loss, prog_bar=True, logger=True)

        return {'loss': loss, 'y': y, 'y_pred': y_pred}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred.to(self.device), y.to(self.device))

        acc = accuracy(y_pred, y)
        acc_weighted = accuracy(y_pred, y, class_reduction='weighted')
        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_acc_weighted", acc_weighted, prog_bar=True,
                 logger=True)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {'loss': loss, 'y': y, 'y_pred': y_pred}

    def validation_epoch_end(self, outputs):
        for i, out in enumerate(outputs):
            y, y_pred = out['y'], out['y_pred']

            if i == 0:
                all_y = y
                all_ypred = y_pred
            else:
                all_y = torch.cat((all_y, y), 0)
                all_ypred = torch.cat((all_ypred, y_pred), 0)

        # from one-hot to labels
        all_ypred = torch.argmax(all_ypred, dim=1).cpu().detach().numpy()
        all_y = all_y.cpu().detach().numpy()

        # plot confusion matrix and log to neptune
        if isinstance(self.logger, NeptuneLogger):
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_confusion_matrix(all_y, all_ypred, ax=ax)
            self.logger.experiment.log_image('confusion_matrix', fig)
            self.logger.experiment.log_text(
                "classification_report",
                str(classification_report(all_y, all_ypred)))

        if self.current_epoch >= self.pseudolabelling_start:
            if self.pseudolabel_mode == 'per_epoch':
                self.pseudolabelling_update_outputs()
                self.pseudolabelling_update_loss()
            elif self.pseudolabel_mode == 'per_batch':
                self.pseudolabelling_update_per_batch()
            else:
                sys.exit(f'Possible modes are "per_batch" and "per_epoch". '
                         f'You assigned {self.pseudolabel_mode}')

    def pseudolabelling_update_loss(self):
        print('Calculating loss for pseudo-labelling')
        optimizer = self.optimizers()
        for i, batch in tqdm(enumerate(self.pseudoloader)):
            output = self.training_step(batch, i, pseudo_label=True)
            loss = output['loss']
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.log("pseudo_loss", loss,
                     prog_bar=True, logger=True)

    def pseudolabelling_update_outputs(self, batch=None, batch_idx=None):
        print('Updating outputs for pseudolabelling')
        if batch is None or batch_idx is None:
            print('Updating outputs for pseudolabelling')
            # update predictions for all batches
            for batch_idx, batch in tqdm(enumerate(self.pseudoloader)):
                output = self.training_step(batch, batch_idx,
                                            pseudo_label=True)
                y_pred = output['y_pred']
                # apply new targets
                for idx, y in enumerate(y_pred):
                    self.pseudoloader.dataset.targets[
                        batch_idx*len(y_pred)+idx] = torch.argmax(y, dim=0)
        else:
            # update predictions for single batch
            output = self.training_step(batch, batch_idx, pseudo_label=True)
            y_pred = output['y_pred']
            # apply new targets
            for idx, y in enumerate(y_pred):
                self.pseudoloader.dataset.targets[batch_idx * len(y_pred) + idx
                                                  ] = torch.argmax(y, dim=0)

        # now switch class_to_idx and classes in pseudo-dataset
        # to the same as in traindataset
        self.pseudoloader.dataset.class_to_idx = {'background': 0, 'bio': 1,
                                                  'glass': 2,
                                                  'metals_and_plastic': 3,
                                                  'non_recyclable': 4,
                                                  'other': 5,
                                                  'paper': 6,
                                                  'unknown': 7}
        self.pseudoloader.dataset.classes = ['background', 'bio', 'glass',
                                             'metals_and_plastic',
                                             'non_recyclable', 'other',
                                             'paper', 'unknown']

    def pseudolabelling_update_per_batch(self):
        optimizer = self.optimizers()
        for batch_idx, batch in tqdm(enumerate(self.pseudoloader)):
            self.pseudolabelling_update_outputs(batch, batch_idx)
            output = self.training_step(batch, batch_idx, pseudo_label=True)
            loss = output['loss']
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.log("pseudo_loss", loss,
                     prog_bar=True, logger=True)
