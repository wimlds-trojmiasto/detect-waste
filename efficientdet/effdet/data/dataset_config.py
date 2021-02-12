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
    
@dataclass
class DetectwasteCfg(CocoCfg):
    variant: str = '2017'
    num_classes: int = 7
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/annotations/annotations_train.json', img_dir='/dih4/dih4_2/wimlds/data/', has_labels=True),
        val=dict(ann_filename='/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/annotations/annotations_test.json', img_dir='/dih4/dih4_2/wimlds/data/', has_labels=True),
    ))
    
@dataclass
class BinaryCfg(CocoCfg):
    variant: str = '2017'
    num_classes: int = 1
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/annotations/annotations_binary_train.json', img_dir='/dih4/dih4_2/wimlds/data/', has_labels=True),
        val=dict(ann_filename='/dih4/dih4_2/wimlds/amikolajczyk/detect-waste/annotations/annotations_binary_test.json', img_dir='/dih4/dih4_2/wimlds/data/', has_labels=True),
    ))


@dataclass
class BinaryMultiCfg(CocoCfg):
    root: str = ""
    ann: str = ""
    variant: str = '2017'
    num_classes: int = 1

    def add_split(self):
        self.splits = {
            'train': {'ann_filename': self.ann+'_train.json',
                      'img_dir': self.root,
                      'has_labels': True},
            'val': {'ann_filename': self.ann+'_test.json',
                    'img_dir': self.root,
                    'has_labels': True}
            }


@dataclass
class TrashCanCfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations0_train.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
        val=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations0_test.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
    ))

@dataclass
class UAVVasteCfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations1_train.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
        val=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations1_test.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
    ))

@dataclass
class ICRACfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations5_train.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
        val=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations5_test.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
    ))

@dataclass
class DrinkWasteCfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations2_train.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
        val=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations2_test.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
    ))

@dataclass
class MJU_WasteCfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations3_train.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
        val=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations3_test.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
    ))
@dataclass
class WadeCfg(CocoCfg):
    variant: str = '2017'
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations4_train.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
        val=dict(ann_filename='/dih4/dih4_home/smajchrowska/detect-waste/annotations/annotations4_test.json', img_dir='/dih4/dih4_2/wimlds/data/all_detect_images', has_labels=True),
    ))
