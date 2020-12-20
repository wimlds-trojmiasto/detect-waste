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
class TACOCfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        val=dict(ann_filename='/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/annotations/annotations_detectwaste.json', img_dir='/dih4/dih4_2/wimlds/TACO-master/data/', has_labels=True),
        train=dict(ann_filename='/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/annotations/annotations-epi.json', img_dir='/dih4/dih4_2/wimlds/data/', has_labels=True),
        #test=dict(ann_filename='annotations/image_info_test2017.json', img_dir='test2017', has_labels=False),
        #testdev=dict(ann_filename='annotations/image_info_test-dev2017.json', img_dir='test2017', has_labels=False),
    ))