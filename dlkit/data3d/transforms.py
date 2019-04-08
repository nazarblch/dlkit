import numpy as np
import torch
import torchvision.transforms
from dlkit.data3d import functional as F


Compose = torchvision.transforms.Compose


class Resize:

    def __init__(self, size):
        self.size = np.asarray(size)

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        zoom_factor = self.size / image.shape
        return {
            **sample,
            'image': F.zoom(image, zoom_factor, order=3),
            'label': None if label is None else F.zoom(label, zoom_factor, order=0),
            '_resolution': sample['_resolution'] / zoom_factor,
        }


class SqueezeToInterval:

    def __init__(self, range=(-1, 1), clip=None):
        self.range = range
        self.clip = clip

    def __call__(self, sample):
        img = sample['image']
        if self.clip is None:
            clip = (img.min(), img.max())
        else:
            clip = self.clip
            img = img.clip(*clip)
        img = (img - clip[0]) / (clip[1] - clip[0]) * (self.range[1] - self.range[0]) + self.range[0]
        return {
            **sample,
            'image': img,
            'label': sample['label'],
        }


class Standardize:

    def __call__(self, sample):
        img = sample['image']
        return {
            **sample,
            'image': (img - img.mean()) / img.std(),
        }


class ToTensor:

    def __call__(self, sample):
        label = sample['label']
        image = sample['image']
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        return {
            **sample,
            'image': torch.from_numpy(image),
            'label': None if label is None else torch.from_numpy(label),
        }


class EncodeOneHot:

    def __init__(self, n_classes=None):
        self.n_classes = n_classes

    def __call__(self, sample):
        label = sample['label']
        if label is None:
            return sample
        if self.n_classes is None:
            n_classes = sample['n_classes']
        else:
            n_classes = self.n_classes
        label = np.eye(n_classes, dtype='float32')[label].transpose(3, 0, 1, 2)
        return {
            **sample,
            'label': label,
        }


class Litter:
    """Randomly generates artifacts on segmentation mask.

    """

    def __init__(self, labels=(1, 2), labels_prob=(0.5, 0.5), prob=0.7, size=0.3, seed=None, empty_val=-1,
                 min_diff=10, alpha=700, sigma=5):
        self.random_state = np.random.RandomState(seed)
        self.prob = prob
        self.labels = labels
        self.labels_prob = labels_prob
        self.empty_val = empty_val
        self.size = size
        self.min_diff = min_diff
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, sample):
        original_mask = sample['label']

        while True:
            mask = np.full_like(original_mask, self.empty_val)

            num_objects = self.random_state.geometric(self.prob)
            for _ in range(num_objects):
                label = self.random_state.choice(self.labels, p=self.labels_prob)
                box = []
                for dim in mask.shape:
                    center = self.random_state.randint(0, dim)
                    half_size = self.random_state.randint(0, dim * self.size) // 2
                    box.append(slice(center - half_size, center + half_size))
                mask[box] = label

            mask = F.label_elastic_transform(mask, self.alpha, self.sigma, self.random_state)

            mask = np.where(mask == self.empty_val, original_mask, mask)

            if (mask != original_mask).sum() >= self.min_diff:
                break

        return {
            **sample,
            'label': mask,
        }


class Choice:
    """Randomly chooses between specified transforms on each call.

    """

    def __init__(self, transforms, probs=None, seed=None):
        self.transforms = transforms
        if probs is None:
            probs = (1. / len(transforms),) * len(transforms)
        self.probs = probs
        self.random_state = np.random.RandomState(seed)

    def __call__(self, sample):
        transform = self.random_state.choice(self.transforms, p=self.probs)
        return transform(sample)


class LabelElasticTransform:
    """Perform elastic transform on segmentation mask only.

    Requires the mask to be encoded in numeric format (not one-hot).

    """

    def __init__(self, alpha, sigma, seed=None):
        self.random_state = np.random.RandomState(seed)
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, sample):
        label = sample['label']

        if isinstance(self.alpha, tuple):
            alpha = self.random_state.uniform(*self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, tuple):
            sigma = self.random_state.uniform(*self.sigma)
        else:
            sigma = self.sigma

        label = F.label_elastic_transform(label, alpha, sigma, self.random_state)

        return {
            **sample,
            'label': label,
        }
