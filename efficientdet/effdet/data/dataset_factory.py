""" Dataset factory

Updated 2021 Wimlds in Detect Waste in Pomerania
"""
from collections import OrderedDict
from pathlib import Path

from .dataset_config import *
from .parsers import *
from .dataset import DetectionDatset
from .parsers import create_parser

# list of detect-waste datasets
waste_datasets_list = ['taco', 'detectwaste', 'binary', 'multi',
                       'uav', 'mju', 'trashcan', 'wade', 'icra'
                       'drinkwaste']


def create_dataset(name, root, ann, splits=('train', 'val')):
    if isinstance(splits, str):
        splits = (splits,)
    name = name.lower()
    root = Path(root)
    dataset_cls = DetectionDatset
    datasets = OrderedDict()
    if name.startswith('coco'):
        if 'coco2014' in name:
            dataset_cfg = Coco2014Cfg()
        else:
            dataset_cfg = Coco2017Cfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                data_dir=root / Path(split_cfg['img_dir']),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )
        datasets = OrderedDict()
    elif name in waste_datasets_list:
        if name.startswith('taco'):
            dataset_cfg = TACOCfg(root=root, ann=ann)
        elif name.startswith('detectwaste'):
            dataset_cfg = DetectwasteCfg(root=root, ann=ann)
        elif name.startswith('binary'):
            dataset_cfg = BinaryCfg(root=root, ann=ann)
        elif name.startswith('multi'):
            dataset_cfg = BinaryMultiCfg(root=root, ann=ann)
            dataset_cfg.add_split(root=root, ann=ann)
        elif name.startswith('uav'):
            dataset_cfg = UAVVasteCfg(root=root, ann=ann)
        elif name.startswith('trashcan'):
            dataset_cfg = TrashCanCfg(root=root, ann=ann)
        elif name.startswith('drinkwaste'):
            dataset_cfg = DrinkWasteCfg(root=root, ann=ann)
        elif name.startswith('mju'):
            dataset_cfg = MJU_WasteCfg(root=root, ann=ann)
        elif name.startswith('wade'):
            dataset_cfg = WadeCfg(root=root, ann=ann)
        elif name.startswith('icra'):
            dataset_cfg = ICRACfg(root=root, ann=ann)
        else:
            assert False, f'Unknown dataset parser ({name})'
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                data_dir=root / Path(split_cfg['img_dir']),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )
    else:
        assert False, f'Unknown dataset parser ({name})'

    datasets = list(datasets.values())
    return datasets if len(datasets) > 1 else datasets[0]
