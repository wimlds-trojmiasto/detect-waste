""" COCO detect-waste dataset configurations

Updated 2021 Wimlds in Detect Waste in Pomerania
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class CocoCfg:
    variant: str = None
    parser: str = 'coco'
    num_classes: int = 80
    splits: Dict[str, dict] = None


@dataclass
class TACOCfg(CocoCfg):
    root: str = ""
    ann: str = ""
    variant: str = '2017'
    num_classes: int = 28

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
class DetectwasteCfg(CocoCfg):
    root: str = ""
    ann: str = ""
    variant: str = '2017'
    num_classes: int = 7

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
class BinaryCfg(CocoCfg):
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
    root: str = ""
    ann: str = ""
    variant: str = '2017'
    num_classes: int = 8

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
class UAVVasteCfg(CocoCfg):
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
class ICRACfg(CocoCfg):
    root: str = ""
    ann: str = ""
    variant: str = '2017'
    num_classes: int = 7

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
class DrinkWasteCfg(CocoCfg):
    root: str = ""
    ann: str = ""
    variant: str = '2017'
    num_classes: int = 4

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
class MJU_WasteCfg(CocoCfg):
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
class WadeCfg(CocoCfg):
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
