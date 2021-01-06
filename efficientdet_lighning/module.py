from types import SimpleNamespace

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from timm.optim import create_optimizer

from efficientdet_lighning.dataloader import create_datasets_and_loaders
from efficientdet_lighning.efficientdet.factory import create_model


class EfficientDetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model(
            model_name='tf_efficientdet_d2',
            bench_task='train',
            num_classes=7,
            pretrained=True,
            pretrained_backbone=True,
            redundant_bias=None,
            label_smoothing=None,
            new_focal=None,
            jit_loss=None,
            bench_labeler=None,
            checkpoint_path='',
        )
        self.model_config = self.model.config
        self.loader_train, self.loader_eval, self.evaluator = create_datasets_and_loaders({}, self.model_config)

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input, target)
        loss = output['loss']
        self.log('train/loss', loss)
        # TO DO model_ema
        return {'loss': loss}

    def train_dataloader(self):
        return self.loader_train

    def configure_optimizers(self, ):
        optim_parameters = {
            'lr': 0.001,
            'opt': 'momentum',
            'opt-eps': 1e-3,
            'momentum': 0.9,
            'weight_decay': 4e-5
        }
        optim_parameters = SimpleNamespace(**optim_parameters)
        optimizer = create_optimizer(optim_parameters, self.model)
        return optimizer


if __name__ == '__main__':
    neptune_logger = NeptuneLogger(
        project_name='detectwaste/efficientdet-lighning',
        experiment_name='effdet-lighning',
    )
    module = EfficientDetModule()
    trainer = pl.Trainer(gpus=[7], gradient_clip_val=10, logger=neptune_logger)
    trainer.fit(module)
