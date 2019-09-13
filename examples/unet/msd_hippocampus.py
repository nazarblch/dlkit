import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


from data_loader import DataLoader
from data_loader.data3d import transforms, datasets

TASK = '/home/nazar/Downloads/Task06_Lung'
NUM_EPOCHS = 250
BATCH_SIZE = 12
IMAGE_SIZE = (48, 64, 48)
LR = 1e-4
VAL_FREQ = 25  # iterations
VAL_SPLIT = dict(num_splits=10, split_num=0, seed=19031)
NSD_TOLERANCE = 4
NUMPY_SEED = 23457
TORCH_SEED = 30117
TEST = False


def init():
    # this is due to bugs in CuDNN, as for April 9, 2019
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    np.random.seed(NUMPY_SEED)
    torch.manual_seed(TORCH_SEED)


class UNet3d(nn.Module):

    def __init__(self, n_classes, activation=nn.ReLU, pooling=nn.MaxPool3d):
        super().__init__()
        self.down0 = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            activation(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            activation(),
        )
        self.down1 = nn.Sequential(
            pooling(2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            activation(),
            nn.Conv3d(128, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            activation(),
        )
        self.down2 = nn.Sequential(
            pooling(2),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            activation(),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            activation(),
        )
        self.down3 = nn.Sequential(
            pooling(2),
            nn.Conv3d(256, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            activation(),
            nn.Conv3d(512, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            activation(),
        )
        self.down4 = nn.Sequential(
            pooling(2),
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.BatchNorm3d(1024),
            activation(),
            nn.Conv3d(1024, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            activation(),
        )
        self.up3 = nn.Sequential(
            nn.Conv3d(1024, 512, 3, padding=1),
            nn.BatchNorm3d(512),
            activation(),
            nn.Conv3d(512, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            activation(),
        )
        self.up2 = nn.Sequential(
            nn.Conv3d(512, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            activation(),
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            activation(),
        )
        self.up1 = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            activation(),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            activation(),
        )
        self.up0 = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            activation(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            activation(),
            nn.Conv3d(64, n_classes, 1),
        )

    def forward(self, inputs):
        out0 = self.down0(inputs)
        out1 = self.down1(out0)
        out2 = self.down2(out1)
        out3 = self.down3(out2)
        out = self.down4(out3)

        out = torch.cat((out3, F.interpolate(out, size=out3.shape[-3:], mode='trilinear', align_corners=False)), dim=1)
        out = self.up3(out)

        out = torch.cat((out2, F.interpolate(out, size=out2.shape[-3:], mode='trilinear', align_corners=False)), dim=1)
        out = self.up2(out)

        out = torch.cat((out1, F.interpolate(out, size=out1.shape[-3:], mode='trilinear', align_corners=False)), dim=1)
        out = self.up1(out)

        out = torch.cat((out0, F.interpolate(out, size=out0.shape[-3:], mode='trilinear', align_corners=False)), dim=1)
        out = self.up0(out)

        return out


def main():
    # writer = Writer('runs/unet/msd_hippocampus', test=TEST)

    transform = transforms.Compose([
        transforms.Standardize(),
        transforms.Resize(IMAGE_SIZE),
        transforms.SqueezeToInterval(clip=(-3, 3)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MSD(root=TASK, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = UNet3d(train_dataset.n_classes)
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    iteration = 0
    best_val_loss = 99999.
    for epoch in range(1, NUM_EPOCHS + 1):
        # print(framework.util.cuda_memory_use())
        # print('Epoch ', epoch, '/', NUM_EPOCHS, writer.dirname)
        for sample in tqdm.tqdm(train_loader):
            iteration += 1
            print(iteration)
            img = sample['image'].cuda()
            lbl = sample['label'].cuda().long()

            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(input=out, target=lbl)
            loss.backward()
            optimizer.step()

            if iteration % VAL_FREQ == 0:
                model.eval()
                model.train()


if __name__ == '__main__':
    init()
    main()
