""" Detection dataset

Hacked together by Ross Wightman
"""
import torch.utils.data as data
import numpy as np
import albumentations as A
import torch

from PIL import Image
from .parsers import create_parser


class DetectionDatset(data.Dataset):
    """`Object Detection Dataset. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, data_dir, parser=None, parser_kwargs=None, transform=None, transforms=None):
        super(DetectionDatset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.data_dir = data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform
        self._transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)
        img_path = self.data_dir / img_info['file_name']
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = torch.as_tensor(np.array(img), dtype=torch.uint8)
            voc_boxes = []
            for coord in target['bbox']:
                xmin = coord[1]
                ymin = coord[0]
                xmax = coord[3]
                ymax = coord[2]
                if xmin<1:
                    xmin = 1
                if ymin<1:
                    ymin = 1
                if xmax>=img.shape[1]-1:
                    xmax = img.shape[1]-1
                if ymax>=img.shape[0]-1:
                    ymax = img.shape[0]-1
                voc_boxes.append([xmin, ymin, xmax, ymax])
            transformed = self.transforms(image=np.array(img), bbox_classes=target['cls'], bboxes=voc_boxes)
            img = torch.as_tensor(transformed['image'], dtype=torch.uint8)
            target['bbox'] = []
            for coord in transformed['bboxes']:
                ymin = int(coord[1])
                xmin = int(coord[0])
                ymax = int(coord[3])
                xmax = int(coord[2])
                target['bbox'].append([ymin, xmin, ymax, xmax])
            target['bbox'] = np.array(target['bbox'], dtype=np.float32)
            target['cls'] = np.array(transformed['bbox_classes'])
            img = Image.fromarray(np.array(img).astype('uint8'), 'RGB')
            target['img_size'] = img.size
            
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, t):
        self._transforms = t

class SkipSubset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        n (int): skip rate (select every nth)
    """
    def __init__(self, dataset, n=2):
        self.dataset = dataset
        assert n >= 1
        self.indices = np.arange(len(dataset))[::n]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    @property
    def parser(self):
        return self.dataset.parser

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, t):
        self.dataset.transform = t

    @property
    def transforms(self):
        return self.dataset.transforms

    @transforms.setter
    def transforms(self, t):
        self.dataset.transforms = t
