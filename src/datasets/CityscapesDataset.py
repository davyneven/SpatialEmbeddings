"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import relabel_sequential

import torch
from torch.utils.data import Dataset


class CityscapesDataset(Dataset):

    class_names = ('person', 'rider', 'car', 'truck',
                   'bus', 'train', 'motorcycle', 'bicycle')
    class_ids = (24, 25, 26, 27, 28, 31, 32, 33)

    def __init__(self, root_dir='./', type="train", class_id=26, size=None, transform=None):

        print('Cityscapes Dataset created')

        # get image and instance list
        image_list = glob.glob(os.path.join(root_dir, 'leftImg8bit/{}/'.format(type), '*/*.png'))
        image_list.sort()
        self.image_list = image_list

        instance_list = glob.glob(os.path.join(root_dir, 'gtFine/{}/'.format(type), '*/*instanceIds*.png'))
        instance_list.sort()
        self.instance_list = instance_list

        self.class_id = class_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):

        index = index if self.size is None else random.randint(0, self.real_size-1)
        sample = {}

        # load image
        image = Image.open(self.image_list[index])
        sample['image'] = image
        sample['im_name'] = self.image_list[index]

        # load instances
        instance = Image.open(self.instance_list[index])
        instance, label = self.decode_instance(instance, self.class_id)
        sample['instance'] = instance
        sample['label'] = label

        # transform
        if(self.transform is not None):
            return self.transform(sample)
        else:
            return sample

    @classmethod
    def decode_instance(cls, pic, class_id=None):
        pic = np.array(pic, copy=False)

        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        if class_id is not None:
            mask = np.logical_and(pic >= class_id * 1000, pic < (class_id + 1) * 1000)
            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(pic[mask])
                instance_map[mask] = ids
                class_map[mask] = 1
        else:
            for i, c in enumerate(cls.class_ids):
                mask = np.logical_and(pic >= c * 1000, pic < (c + 1) * 1000)
                if mask.sum() > 0:
                    ids, _, _ = relabel_sequential(pic[mask])
                    instance_map[mask] = ids + np.amax(instance_map)
                    class_map[mask] = i+1

        return Image.fromarray(instance_map), Image.fromarray(class_map)
