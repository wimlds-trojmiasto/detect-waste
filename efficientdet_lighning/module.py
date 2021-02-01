from types import SimpleNamespace

import pytorch_lightning as pl
from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import NeptuneLogger
from timm.optim import create_optimizer

from efficientdet_lighning.dataloader import create_datasets_and_loaders
from efficientdet_lighning.efficientdet.factory import create_model


class EfficientDetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model(
            model_name='tf_efficientdet_d2', #yamlconfig
            bench_task='train',
            num_classes=7,
            pretrained=True, #yamlconfig
            pretrained_backbone=True, #yamlconfig
            redundant_bias=None,
            label_smoothing=None,
            new_focal=None,
            jit_loss=None,
            bench_labeler=None,
            checkpoint_path='',
        )
        self.model_config = self.model.config

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input, target)
        loss = output['loss']
        self.log('train/loss', loss, sync_dist=True)
        # TO DO model_ema
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input, target)
        loss = output['loss']
        self.log('val/loss', loss, sync_dist=True)
        return {'val_loss': loss,
                'detections': output['detections'],
                'target': target}

    def validation_epoch_end(self, outputs):
        for batch in outputs:
            self.evaluator.add_predictions(batch['detections'], batch['target'])
        metrics = self.evaluator.evaluate()
        for key, metric in metrics.items():
            metric_name = key.split('/')[-1]
            metric_name = f'valid/mAP/{metric_name}'
            self.log(metric_name, metric, sync_dist=True)

    def train_dataloader(self):
        self.loader_train, _, _ = create_datasets_and_loaders({}, self.model_config)
        return self.loader_train

    def val_dataloader(self):
        _, self.loader_eval, self.evaluator = create_datasets_and_loaders({}, self.model_config)
        return self.loader_eval

    def configure_optimizers(self, ):
        optim_parameters = {
            'lr': 0.001, #yamlconfig
            'opt': 'momentum', #yamlconfig
            'opt-eps': 1e-3, #yamlconfig
            'momentum': 0.9, #yamlconfig
            'weight_decay': 4e-5 #yamlconfig
        }
        optim_parameters = SimpleNamespace(**optim_parameters)
        optimizer = create_optimizer(optim_parameters, self.model)
        return optimizer


if __name__ == '__main__':
    neptune_logger = NeptuneLogger(
        project_name='detectwaste/efficientdet-lighning', #yamlconfig
        experiment_name='effdet-lighning', #yamlconfig
    )
    gpu_monitor = GPUStatsMonitor()
    lr_logger = LearningRateMonitor()
    module = EfficientDetModule()
    trainer = pl.Trainer(gpus=[0, 1, 2, 3, 4, 5, 6, 7], #yamlconfig
                         accelerator='ddp',
                         replace_sampler_ddp=False,
                         gradient_clip_val=10, #yamlconfig
                         logger=neptune_logger,
                         # limit_train_batches=4 * 4 * 12,
                         # limit_val_batches=0,
                         log_every_n_steps=10,
                         sync_batchnorm=True,
                         max_epochs=50, #yamlconfig
                         callbacks=[gpu_monitor, lr_logger]
                         )

    trainer.fit(module)
