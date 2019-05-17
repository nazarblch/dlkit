import os
import json
import numpy as np
import nibabel

import torch.utils.data

from data_loader.data3d import functional

COLORS = (
    ( 64,  64, 64 ),
    (255, 255, 0  ),
    (255, 128, 128)
)


class MSD(torch.utils.data.Dataset):

    def __init__(self, root, test=False, resolution=1, transform=None, ignore_labels=False, preserve_original=False,
                 store_index=False, store_sample=False):
        """PyTorch dataset for Medical Segmentation Decathlon data_loader.

        Loads each sample into two NumPy arrays. Image is stored
        under the key `image` and segmentation mask is stored under
        the key `label`.

        Args:
            root: path to one of the MSD tasks
            test: load test set instead of train
            resolution: resolution in mm
            transform: PyTorch transform
            ignore_labels: do not load labels
            preserve_original: store the original label along with
                the original resolution in sample's meta under the keys
                `_original_label` and `_original_resolution`
            store_index: store sample's index in meta under
                the key`_id`
            store_sample: store sample's paths in meta under
                the key`_sample`
        """
        self.root = root
        self.test = test
        self.resolution = resolution
        self.transform = transform
        self.ignore_labels = ignore_labels
        self.preserve_original = preserve_original
        self.store_index = store_index
        self.store_sample = store_sample
        with open(os.path.join(root, 'dataset.json'), 'r') as f:
            self.meta = json.load(f)
        if test:
            self.samples = self.meta['test']
        else:
            self.samples = self.meta['training']

    def load_nii(self, index):
        sample = self.samples[index]
        if self.test:
            filename = os.path.join(self.root, sample)
            image = nibabel.load(filename)
            label = None
        else:
            filename = os.path.join(self.root, sample['image'])
            image = nibabel.load(filename)
            if self.ignore_labels:
                label = None
            else:
                filename = os.path.join(self.root, sample['label'])
                label = nibabel.load(filename)
                assert image.shape[:3] == label.shape[:3]
                assert image.header.get_xyzt_units()[0] == label.header.get_xyzt_units()[0] == 'mm'
                assert (image.affine == label.affine).all()
        output = {
            'image': image,
            'label': label,
        }
        if self.store_index:
            output['_id'] = index
        if self.store_sample:
            output['_sample'] = sample
        return output

    def load_numpy(self, index):
        sample = self.load_nii(index)
        image = sample['image']
        label = sample['label']

        actual_resolution = abs(image.affine.dot((1, 1, 1, 0))[:3])
        required_resolution = np.asarray((self.resolution,) * 3)
        zoom_factor = tuple(actual_resolution / required_resolution)

        image = image.get_data().astype('float32')
        assert image.ndim in (3, 4)
        if image.ndim == 4:
            image = image.transpose(3, 0, 1, 2)
        sample['image'] = functional.zoom(image, zoom_factor, order=3)

        if label is not None:
            label = label.get_data().astype('int32')
            assert label.ndim == 3
            sample['label'] = functional.zoom(label, zoom_factor, order=0)

        sample['_resolution'] = required_resolution
        if self.preserve_original:
            sample['_original_resolution'] = required_resolution
            sample['_original_label'] = sample['label']

        return sample

    def __getitem__(self, index):
        sample = self.load_numpy(index)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)

    @property
    def n_classes(self):
        return len(self.meta['labels'])

    @property
    def classes(self):
        return [(int(key), val, COLORS[i]) for i, (key, val) in enumerate(self.meta['labels'].items())]

    @staticmethod
    def get_classes(root):
        with open(os.path.join(root, 'dataset.json'), 'r') as f:
            meta = json.load(f)
        return meta['labels']
