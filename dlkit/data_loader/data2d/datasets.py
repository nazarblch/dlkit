import os
from PIL import Image
import torch.utils.data
import torchvision.datasets
import numpy as np


class Cityscapes(torchvision.datasets.Cityscapes):

    classes = [
        ( 7, "road",          (128,  64, 128)),
        ( 8, "sidewalk",      (244,  35, 232)),
        (11, "building",      ( 70,  70,  70)),
        (12, "wall",          (102, 102, 156)),
        (13, "fence",         (190, 153, 153)),
        (17, "pole",          (153, 153, 153)),
        (19, "traffic_light", (250, 170,  30)),
        (20, "traffic_sign",  (220, 220,   0)),
        (21, "vegetation",    (107, 142,  35)),
        (22, "terrain",       (152, 251, 152)),
        (23, "sky",           (  0, 130, 180)),
        (24, "person",        (220,  20,  60)),
        (25, "rider",         (255,   0,   0)),
        (26, "car",           (  0,   0, 142)),
        (27, "truck",         (  0,   0,  70)),
        (28, "bus",           (  0,  60, 100)),
        (31, "train",         (  0,  80, 100)),
        (32, "motorcycle",    (  0,   0, 230)),
        (33, "bicycle",       (119,  11,  32)),
    ]

    n_classes = len(classes)

    def __init__(self, root, split='train', mode='gtFine', target_type='instance',
                 transform=None, target_transform=None, joint_transform=None):
        super().__init__(root, split, mode, target_type, transform, target_transform)
        self.joint_transform = joint_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.joint_transform is not None:
            image, target = self.joint_transform(image, target)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target


class VOCSegmentation(torch.utils.data.Dataset):

    classes = [
        ( 0, 'background',  (  0,   0,   0)),
        ( 1, 'aeroplane',   (128,   0,   0)),
        ( 2, 'bicycle',     (  0, 128,   0)),
        ( 3, 'bird',        (128, 128,   0)),
        ( 4, 'boat',        (  0,   0, 128)),
        ( 5, 'bottle',      (128,   0, 128)),
        ( 6, 'bus',         (  0, 128, 128)),
        ( 7, 'car',         (128, 128, 128)),
        ( 8, 'cat',         ( 64,   0,   0)),
        ( 9, 'chair',       (192,   0,   0)),
        (10, 'cow',         ( 64, 128,   0)),
        (11, 'diningtable', (192, 128,   0)),
        (12, 'dog',         ( 64,   0, 128)),
        (13, 'horse',       (192,   0, 128)),
        (14, 'motorbike',   ( 64, 128, 128)),
        (15, 'person',      (192, 128, 128)),
        (16, 'pottedplant', (  0,  64,   0)),
        (17, 'sheep',       (128,  64,   0)),
        (18, 'sofa',        (  0, 192,   0)),
        (19, 'train',       (128, 192,   0)),
        (20, 'tvmonitor',   (  0,  64, 128)),
    ]

    n_classes = len(classes)

    # over all pixels, for the three channels
    train_mean = (0.45676398, 0.44254407, 0.40738845)
    train_std = (0.27287834, 0.26932333, 0.28497848)

    def __init__(self,
                 root,
                 image_set='train',
                 augmented=False,
                 transform=None,
                 target_transform=None,
                 joint_transform=None,
                 detect_edges=None):
        year = '2012'
        self.root = os.path.expanduser(root)
        self.year = year
        self.augmented = augmented
        self.filename = torchvision.datasets.voc.DATASET_YEAR_DICT[year]['filename']
        self.md5 = torchvision.datasets.voc.DATASET_YEAR_DICT[year]['md5']
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.detect_edges = detect_edges
        self.image_set = image_set

        if augmented:
            segmentation_class_dir = 'SegmentationClassAug'
            image_sets_dir = 'ImageSets/SegmentationAug'
        else:
            segmentation_class_dir = 'SegmentationClass'
            image_sets_dir = 'ImageSets/Segmentation'

        base_dir = torchvision.datasets.voc.DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, segmentation_class_dir)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(voc_root, image_sets_dir)

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.detect_edges is not None and np.asarray(target).max() < self.n_classes:
            target = self.detect_edges(target)

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)
