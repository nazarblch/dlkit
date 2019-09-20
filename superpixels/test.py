#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init
from torch import Tensor

from data_loader.data2d.dataset.superpixels import Superpixels
from framework.nn.modules.common.sp_pool import SPPoolMean
from framework.nn.modules.resnet.residual import ResidualNet
from framework.segmentation.base import PenalizedSegmentation
from framework.segmentation.fcn import FCNSegmentation
from framework.segmentation.loss.modularity import VGGModularity
from framework.segmentation.loss.sp_loss import SuperPixelsLoss
from framework.segmentation.resnet import ResidualSegmentation
from framework.segmentation.unet import UNetSegmentation
from superpixels import mbs
from superpixels.mbs import superpixels
import albumentations

use_cuda = torch.cuda.is_available()

print(use_cuda)



parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=2, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=5000, type=int,
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=50, type=float,
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
args = parser.parse_args()

image_size = 256


dataset = Superpixels(
    root="/home/nazar/PycharmProjects/segmentation_data/leftImg8bit/train/strasbourg",
    compute_sp=True,
    transforms_al=albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.CenterCrop(image_size, image_size),
    ]),
    transform=transforms.Compose([
        transforms.ToTensor()
    ]),
    target_transform=transforms.ToTensor()
)


segm_net = PenalizedSegmentation(
    FCNSegmentation(100)
)
segm_net.add_penalty(SuperPixelsLoss(1.0))
segm_net.add_penalty(VGGModularity(3, 0.01))


label_colours = np.random.randint(255, size=(100, 3))


for batch_idx in range(args.maxIter):

    im_0, sp_0 = dataset[batch_idx % 3]

    data = im_0.cuda().unsqueeze(0).type(torch.float32)
    sp = sp_0.cuda().type(torch.int64).unsqueeze(0)

    output = segm_net.forward(data, sp)

    im_target = output.max(dim=1)[1].cpu().numpy()
    nLabels = len(np.unique(im_target))

    if args.visualize and batch_idx % 19 == 1:
        im_target_rgb = np.array([label_colours[c%100] for c in im_target])
        im_target_rgb = im_target_rgb.reshape((image_size, image_size, 3)).astype(np.uint8)
        cv2.imshow("output", im_target_rgb)
        cv2.waitKey(20)

    # superpixel refinement
    loss = segm_net.train(data, sp)

    print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item())
    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

