import numpy as np
import cv2
from matplotlib import pyplot as plt
from torch import Tensor
from typing import List

from superpixels import mbspy
from multiprocessing import Pool
import torch


def superpixels(img: np.ndarray) -> np.ndarray:

    assert len(img.shape) == 3
    assert img.shape[2] <= 3

    mat = mbspy.Mat.from_array(img.astype(np.uint8))
    sp = mbspy.superpixels(mat, int(img.shape[1]//2), 0.1)
    sp_nd = np.asarray(sp)
    # sp_nd += 1

    assert sp_nd.shape[0] == img.shape[0]
    assert sp_nd.shape[1] == img.shape[1]

    return sp_nd


proc_pool = Pool(8)


def superpixels_seq(images: List[np.ndarray]) -> List[np.ndarray]:

    sp_seq = proc_pool.map(superpixels, images)
    return sp_seq


def superpixels_nd(tensor: np.ndarray) -> np.ndarray:

    assert len(tensor.shape) == 4
    assert tensor.shape[1] <= 3

    n = tensor.shape[0]

    tensor_t = tensor.transpose((0, 2, 3, 1))

    img_seq = [tensor_t[i] for i in range(n)]

    sp_seq = proc_pool.map(superpixels, img_seq)

    sp_seq = [sp[np.newaxis, np.newaxis, :, :] for sp in sp_seq]

    sp_seq = np.concatenate(sp_seq, axis=0)

    assert tensor.shape[0] == sp_seq.shape[0]
    assert tensor.shape[2] == sp_seq.shape[2]
    assert tensor.shape[3] == sp_seq.shape[3]

    return sp_seq


def superpixels_tensor(image: Tensor) -> Tensor:
    device = image.device
    image = image.detach()
    if image.is_cuda:
        image = image.cpu()

    sp = superpixels_nd(image.numpy())

    return torch.from_numpy(sp).type(torch.int64).to(device)


if __name__ == "__main__":

    img = cv2.imread('/home/nazar/PycharmProjects/segmentation_data/leftImg8bit/train/strasbourg/strasbourg_000000_000065_leftImg8bit.png')
    sp_nd = superpixels(img)

    print(sp_nd)

    plt.imshow(sp_nd)
    plt.show()

