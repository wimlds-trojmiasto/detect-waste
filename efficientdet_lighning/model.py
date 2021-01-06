from efficientdet_lighning.efficientdet.factory import create_model
model = create_model(
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
if __name__ == '__main__':
    print(model)
