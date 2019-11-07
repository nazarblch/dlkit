import numpy as np
import scipy.ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import warnings


def zoom(image, zoom_factor, order=3):
    """Analogous to 2D `scipy.ndimage.zoom`, but for 3D data.

    For 4D inputs the first dimension is considered channels.
    Interpolation is performed for each channel separately.

    Args:
        image: a 3D or 4D tensor
        zoom_factor: int or 3-tuple.
        order: the order of the spline interpolation, same as in the
            original `zoom`

    Returns:
        3D or 4D array
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if image.ndim == 3:
            return scipy.ndimage.zoom(image, zoom_factor, order=order)
        elif image.ndim == 4:
            channels = [
                scipy.ndimage.zoom(image[i, :, :, :], zoom_factor, order=order)
                for i in range(image.shape[0])
            ]
            return np.stack(channels)
        else:
            raise ValueError('this function operates on d3 volumes (possibly with multiple channels) only')


def elastic_transform(image, alpha, sigma, random_state=None, order=0):
    """Elastic transformation.

    Args:
        image: 3D image
        alpha:
        sigma:
        random_state:
        order:

    Returns:
        3D array
    """
    if random_state is None:
        random_state = np.random.RandomState()
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))
    distorted_image = map_coordinates(image, indices, order=order, mode='reflect')
    return distorted_image.reshape(image.shape)
