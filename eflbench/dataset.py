import logging
import random
from shutil import rmtree
from typing import List
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor

from omegaconf import DictConfig
from hydra.utils import call

logger = logging.getLogger("emtbench")
CLASS_PREFIX = 'cls_'

def get_basic_transform():
    normal = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return Compose([ToTensor(), normal])
    
def _get_random_list_of_integers(max_int: int, num_ints: int):
    return [random.randint(0,max_int-1) for _ in range(num_ints)]

def get_inmemory_dummy_dataset(num_images: int, num_classes: int, input_shape: List[int], labels: List[int]=None):
    """Return a very rudimentary list of tensors that can be interpreted as a dataset. """
    logger.info(f"Creatting dummy in-memory dataset")

    # first
    if labels is None:
        labels = _get_random_list_of_integers(num_classes, num_images)

    dataset = [(torch.randn(tuple(input_shape)), lbl) for lbl in labels]
    return dataset

def erase_dataset_fs_dir(path_to_data:str):

    data_dir = Path(path_to_data)
    if data_dir.exists():
        # We erase it and create it again. This is the easiest way of supporting you running different "datasets"
        rmtree(str(data_dir))


def get_infs_dummy_dataset(path_to_data: str, num_images: int, num_classes: int, input_shape: List[int]):
    """Writes a bunch of images to disk. Then treats them as images from a coherent dataset."""

    # Ensure at one image per class
    assert num_images >= num_images, "The number of images must be >= number of classes." \
        "This is to keep the logic around torchvision's ImageFolder dataset simple"

    # if directory doesn't exist, create and write images
    data_dir = Path(path_to_data)
    if not data_dir.exists():

        logger.info(f"Creatting dummy in-FS dataset")
        data_dir.mkdir(parents=True)
        
        # ensure there is a directory for each class (needed because of how TV's ImageFolder works)
        for lbl in range(num_classes):
            class_dir = data_dir/f"{CLASS_PREFIX}{lbl}"
            class_dir.mkdir()

        # Generated dummy dataset
        # ensure at least one image for each class
        labels = list(range(num_classes))
        labels.extend(_get_random_list_of_integers(num_classes, num_images-num_classes))
        dataset = get_inmemory_dummy_dataset(num_images, num_classes, input_shape, labels=labels)

        # Write images to disk
        for i, (tensor, lbl) in enumerate(dataset):
            class_dir = data_dir/f"{CLASS_PREFIX}{lbl}"
            img = (255. * tensor).permute((2,1,0)).numpy().astype(np.uint8)
            img_name = class_dir/f"{i}.jpeg"
            cv2.imwrite(str(img_name), img)

    # create dataset using native ImageFolder dataset
    dataset = ImageFolder(path_to_data, transform=get_basic_transform())
    return dataset


def get_simple_dataloader(get_dataset_fn: DictConfig, batch_size: int, num_workers:int):

    # get dataset
    dataset = call(get_dataset_fn)

    # Then prepare a dataloader 
    dloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    return dloader
