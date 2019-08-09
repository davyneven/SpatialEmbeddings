"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm


def process(tup):
    im, inst = tup

    image_path = os.path.splitext(os.path.relpath(im, os.path.join(IMAGE_DIR, 'train')))[0]
    image_path = os.path.join(IMAGE_DIR, 'crops', image_path)
    instance_path = os.path.splitext(os.path.relpath(inst, os.path.join(INSTANCE_DIR, 'train')))[0]
    instance_path = os.path.join(INSTANCE_DIR, 'crops', instance_path)

    try:  # can't use 'exists' because of threads
        os.makedirs(os.path.dirname(image_path))
        os.makedirs(os.path.dirname(instance_path))
    except FileExistsError:
        pass

    image = Image.open(im)
    instance = Image.open(inst)
    w, h = image.size

    instance_np = np.array(instance, copy=False)
    object_mask = np.logical_and(instance_np >= OBJ_ID * 1000, instance_np < (OBJ_ID + 1) * 1000)
    
    ids = np.unique(instance_np[object_mask])
    ids = ids[ids!= 0]

    # loop over instances
    for j, id in enumerate(ids):
        
        y, x = np.where(instance_np == id)
        ym, xm = np.mean(y), np.mean(x)
        
        ii = int(np.clip(ym-CROP_SIZE/2, 0, h-CROP_SIZE))
        jj = int(np.clip(xm-CROP_SIZE/2, 0, w-CROP_SIZE))

        im_crop = image.crop((jj, ii, jj + CROP_SIZE, ii + CROP_SIZE))
        instance_crop = instance.crop((jj, ii, jj + CROP_SIZE, ii + CROP_SIZE))

        im_crop.save(image_path + "_{:03d}.png".format(j))
        instance_crop.save(instance_path + "_{:03d}.png".format(j))


if __name__ == '__main__':
    # cityscapes dataset
    CITYSCAPES_DIR=os.environ.get('CITYSCAPES_DIR')

    IMAGE_DIR=os.path.join(CITYSCAPES_DIR, 'leftImg8bit')
    INSTANCE_DIR=os.path.join(CITYSCAPES_DIR, 'gtFine')
    OBJ_ID = 26
    CROP_SIZE=512

    # load images/instances
    images = glob.glob(os.path.join(IMAGE_DIR, 'train', '*/*.png'))
    images.sort()
    instances = glob.glob(os.path.join(INSTANCE_DIR, 'train', '*/*instanceIds.png'))
    instances.sort()

    with Pool(8) as p:
        r = list(tqdm(p.imap(process, zip(images,instances)), total=len(images)))
