import json
import os
import shutil
from collections import namedtuple
import zipfile
from multiprocessing import Pool
from typing import Tuple

import numpy as np
from PIL import Image
import cv2
from torch import Tensor
from torchvision.datasets import VisionDataset
from superpixels.mbs import superpixels


class Superpixels(VisionDataset):

    def __init__(
            self,
            root,
            target_root,
            compute_sp=True,
            transform=None,
            target_transform=None,
            transforms=None,
            transforms_al=None
    ):
        super(Superpixels, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = self.root
        self.targets_dir = target_root
        self.images = []
        self.targets = []
        self.transforms_al = transforms_al

        if not os.path.isdir(self.images_dir):
            raise RuntimeError('Dataset not found or incomplete.')

        if not os.path.isdir(self.targets_dir) and compute_sp is False:
            raise RuntimeError('Super pixels dataset not found or incomplete.')

        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            self.targets.append(os.path.join(self.targets_dir, file_name) + ".npy")

        def compute_superpixels_one(im_path, target_path):
            img = cv2.imread(im_path)
            sp = superpixels(img)
            np.save(target_path, sp)

        if compute_sp:
            if os.path.exists(self.targets_dir):
                shutil.rmtree(self.targets_dir)
            os.makedirs(self.targets_dir)
            print("Computing super pixels ...")
            for i in range(len(self.images)):
                compute_superpixels_one(self.images[i], self.targets[i])
            print("written into " + self.targets_dir)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:

        image = cv2.imread(self.images[index])
        target = np.load(self.targets[index])

        if self.transforms_al is not None:
            data = {"image": image, "mask": target}
            augmented = self.transforms_al(**data)
            image, target = augmented["image"], augmented["mask"]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)

