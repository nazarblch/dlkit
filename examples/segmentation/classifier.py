from typing import List

import requests
from torchvision.datasets import Cityscapes
from torchvision.models import resnet50
from torchvision.transforms import transforms
import torch
from torch import nn, Tensor

from data_loader.DataPath import DataPath
from framework.loss import Loss
from config import ParallelConfig
from framework.segmentation.Mask import Mask
from segmentation.loss.neighbour_diff import NeighbourDiffLoss
from framework.segmentation.unet import UNetSegmentation
from viz.visualization import show_segmentation, show_images

LABELS_URL = 'https://s3.amazonaws.com/mlpipes/pytorch-quick-start/labels.json'
labels = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}

transform = transforms.Compose([            #[1]
 transforms.Resize(128),                    #[2]
 transforms.CenterCrop(128),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

mask_transform = transforms.Compose([            #[1]
 transforms.Resize(128),                    #[2]
 transforms.CenterCrop(128),                #[3]
 transforms.ToTensor()                 #[4]
 ])


# trainset = CocoStuff10k(root="/home/nazar/PycharmProjects/coco", transform=transform)
trainset = Cityscapes(DataPath.HOME_STREET, transform=transform, target_transform=mask_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=12)


class MaskClassifier(nn.Module):

    def __init__(self):

        super(MaskClassifier, self).__init__()
        self.resnet = resnet50(pretrained=True).to(ParallelConfig.MAIN_DEVICE)
        for param in self.resnet.parameters():
            param.requires_grad = False

    def classify(self, img: Tensor) -> Tensor:
        return self.resnet(img).softmax(dim=1)

    def forward(self, img: Tensor, masks: Tensor) -> Tensor:

        s = masks.sum()
        masks = masks.split(1, dim=1)
        predictions = []
        for m in masks:
            if m.sum() > s * 0.1:
                pred = self.classify(img * m).view(img.size(0), 1, -1)

                predictions.append(pred)

        return torch.cat(predictions, dim=1)

    def loss(self, img: Tensor, masks: Tensor) -> Loss:

        P: Tensor = self.forward(img, masks)
        H = (-P * (P + 1e-8).log()).sum(dim=2).mean()

        loss = H

        for i in range(P.shape[1]):
            for j in range(P.shape[1]):
                if i != j:
                    loss += (P[:, i] * P[:, j]).sum(dim=1).mean()

        return Loss(loss)


labels_list: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

segm_net = torch.nn.DataParallel(
    UNetSegmentation(labels_list.__len__()).to(ParallelConfig.MAIN_DEVICE),
    ParallelConfig.GPU_IDS
)

opt = torch.optim.Adam(segm_net.parameters(), lr=0.0002, betas=(0.5, 0.999))


classifier = MaskClassifier()


print("Starting Training Loop...")
# For each epoch
for epoch in range(5):
    # For each batch in the dataloader
    for i, (imgs, labels) in enumerate(trainloader, 0):

        imgs = imgs.to(ParallelConfig.MAIN_DEVICE)
        segm: Tensor = segm_net(imgs)

        cl_loss = classifier.loss(imgs, segm)
        cl_loss += Loss(segm.mean() / 2)
        cl_loss += NeighbourDiffLoss(5)(Mask(segm)) * 3


        opt.zero_grad()
        cl_loss.minimize()
        opt.step()
        print(cl_loss.item())

        if i % 20 == 0:
            show_images(imgs.cpu(), 2, 2)
            show_segmentation(segm.cpu())
