""" COCO, VOC, OpenImages dataset configurations

Copyright 2020 Ross Wightman
"""
import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CocoCfg:
    variant: str = None
    parser: str = 'coco'
    num_classes: int = 80
    splits: Dict[str, dict] = None

@dataclass
class Coco2017Cfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='annotations/instances_train2017.json', img_dir='train2017', has_labels=True),
        val=dict(ann_filename='annotations/instances_val2017.json', img_dir='val2017', has_labels=True),
        test=dict(ann_filename='annotations/image_info_test2017.json', img_dir='test2017', has_labels=False),
        testdev=dict(ann_filename='annotations/image_info_test-dev2017.json', img_dir='test2017', has_labels=False),
    ))