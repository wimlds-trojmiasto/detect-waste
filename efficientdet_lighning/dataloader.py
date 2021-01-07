import neptune

from efficientdet_lighning.efficientdet.anchors import AnchorLabeler, Anchors
from efficientdet_lighning.efficientdet.data import create_dataset, create_loader
from efficientdet_lighning.efficientdet.data import resolve_input_config, SkipSubset
from efficientdet_lighning.efficientdet.evaluator import create_evaluator
from model import model_config


def create_datasets_and_loaders(args, model_config):
    input_config = resolve_input_config(args, model_config=model_config)

    dataset_train, dataset_eval = create_dataset('DetectwasteCfg', "/dih4/dih4_2/wimlds/data/")

    # setup labeler in loader/collate_fn if not enabled in the model bench
    labeler = None
    bench_labeler = False
    if not bench_labeler:
        labeler = AnchorLabeler(
            Anchors.from_config(model_config), model_config.num_classes, match_threshold=0.5)

    loader_train = create_loader(
        dataset=dataset_train,
        input_size=input_config['input_size'],
        batch_size=4,
        is_training=True,
        use_prefetcher=True,
        re_prob=0.2,
        re_mode='pixel',
        re_count=1,
        # color_jitter=args.color_jitter,
        # auto_augment=args.aa,
        interpolation='random' or input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=4,
        distributed=False,
        pin_mem=False,
        anchor_labeler=labeler,
    )

    val_skip = 0
    if val_skip > 1:
        dataset_eval = SkipSubset(dataset_eval, val_skip)
    loader_eval = create_loader(
        dataset_eval,
        input_size=input_config['input_size'],
        batch_size=4,
        is_training=False,
        use_prefetcher=True,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=4,
        distributed=False,
        pin_mem=False,
        anchor_labeler=labeler,
    )

    evaluator = create_evaluator('DetectwasteCfg', loader_eval.dataset,
                                 distributed=False, pred_yxyx=False)

    return loader_train, loader_eval, evaluator


if __name__ == '__main__':
    args = {}
    output = create_datasets_and_loaders(args, model_config)
    print('sdfsdf')
