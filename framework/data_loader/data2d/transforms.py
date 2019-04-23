from torchvision.transforms import *
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
import numpy as np

from dlkit.data_loader.data2d import functional


class RenameLabels(object):

    def __init__(self, mapping, ignore_index):
        self.mapping = mapping
        self.ignore_index = ignore_index

    def __call__(self, mask):
        mask = np.array(mask, dtype=np.uint8)
        left = np.ones_like(mask, dtype=bool)
        for i, lab in enumerate(self.mapping):
            temp = mask == lab
            left &= ~temp
            mask[temp] = i
        mask[left] = self.ignore_index
        return Image.fromarray(mask, mode='P')


class ToTensorNotScaled(object):

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return functional.to_tensor_not_scaled(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class JointRandomResizedCrop(RandomResizedCrop):

    def __call__(self, img, lbl):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return (
            F.resized_crop(img, i, j, h, w, self.size, self.interpolation),
            F.resized_crop(lbl, i, j, h, w, self.size, Image.NEAREST),
        )


class FactorResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, factor, interpolation=Image.BILINEAR):
        self.factor = factor
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        size = np.ceil(np.asarray(img.size) / self.factor).astype(int)
        return F.resize(img, (size[1], size[0]), self.interpolation)


class DetectEdges(object):

    def __init__(self, size=3, edge_index=255):
        self.size = size
        self.edge_index = edge_index

    def __call__(self, labels):
        edges = (
            np.asarray(labels.filter(ImageFilter.MinFilter(self.size))) !=
            np.asarray(labels.filter(ImageFilter.MaxFilter(self.size)))
        )
        labels = np.array(labels)
        labels[edges] = self.edge_index
        return Image.fromarray(labels, mode='P')
