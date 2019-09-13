import numpy as np
import cv2
from matplotlib import pyplot as plt
from torch import Tensor

from superpixels import mbspy
from multiprocessing import Pool
import torch


def superpixels(img: np.ndarray) -> np.ndarray:

    expand = False
    if img.shape[0] == 1:
        img = img[0]
        expand = True

    mat = mbspy.Mat.from_array(img)
    sp = mbspy.superpixels(mat, img.shape[0], 0.01)
    sp_nd = np.asarray(sp)
    if np.min(sp_nd) < 0:
       sp_nd += 1

    if expand:
        sp_nd = np.expand_dims(sp_nd, 0)

    return sp_nd


if __name__ == "__main__":

    img = cv2.imread('/home/nazar/Downloads/person.jpg')
    sp_nd = superpixels(img)

    print(sp_nd)

    plt.imshow(sp_nd)
    plt.show()

