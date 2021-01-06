def create_datasets_and_loaders(args, model_config):
    input_config = resolve_input_config(args, model_config=model_config)

    dataset_train, dataset_eval = create_dataset(args.dataset, args.root)

    # setup labeler in loader/collate_fn if not enabled in the model bench
    labeler = None
    if not args.bench_labeler:
        labeler = AnchorLabeler(
            Anchors.from_config(model_config), model_config.num_classes, match_threshold=0.5)

    loader_train = create_loader(
        dataset_train,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        # color_jitter=args.color_jitter,
        # auto_augment=args.aa,
        interpolation=args.train_interpolation or input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
    )

    if args.val_skip > 1:
        dataset_eval = SkipSubset(dataset_eval, args.val_skip)
    loader_eval = create_loader(
        dataset_eval,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
    )

    evaluator = create_evaluator(args.dataset, loader_eval.dataset, neptune,
                                 distributed=args.distributed, pred_yxyx=False)

    return loader_train, loader_eval, evaluator
