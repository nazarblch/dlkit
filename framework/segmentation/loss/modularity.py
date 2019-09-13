from framework.Loss import Loss
from framework.nn.modules.common.vgg import Vgg16
from framework.nn.ops.image_pixel_dist import LocalPixelDistance
from framework.nn.ops.pairwise_map import LocalPairwiseMap2D
from torch import Tensor, nn

from framework.parallel import ParallelConfig
from framework.segmentation.Mask import Mask


class Modularity(nn.Module):

    def __init__(self, kernel_size: int, sigma: float):
        super(Modularity, self).__init__()
        self.pixel_pair_weight = LocalPixelDistance(kernel_size, sigma_t=sigma)
        self.prob = LocalPairwiseMap2D(kernel_size)

    def forward(self, images: Tensor, segmentation: Tensor) -> Tensor:

        n = images.size(0)
        K = segmentation.size(1)

        p12 = self.prob(segmentation, lambda di, dj, p1, p2: p1 * p2)
        w12 = self.pixel_pair_weight(images).detach()

        n_pairs = p12.size(3)
        p1 = self.prob(segmentation, lambda di, dj, p1, p2: p1 / n_pairs).sum(dim=3)
        w1 = w12.sum(dim=3).detach()

        rel = (p12 * w12).sum(dim=(1, 3)) / (w1 * p1).sum(dim=1)

        assert rel.shape == (n, K)

        return rel.sum(dim=1) / (K * n)


class MultiLayerModularity(nn.Module):

    def __init__(self, kernel_size: int):
        super(MultiLayerModularity, self).__init__()
        self.kernel_size = kernel_size
        self.vgg = Vgg16(8).to(ParallelConfig.MAIN_DEVICE)
        if ParallelConfig.GPU_IDS.__len__() > 1:
            self.vgg = nn.DataParallel(self.vgg, ParallelConfig.GPU_IDS)

        # self.vgg_2 = Vgg16(16).to(ParallelConfig.MAIN_DEVICE)
        # if ParallelConfig.GPU_IDS.__len__() > 1:
        #     self.vgg_2 = nn.DataParallel(self.vgg_2, ParallelConfig.GPU_IDS)

        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

        self.modularity_img = nn.DataParallel(Modularity(self.kernel_size, sigma=0.2), ParallelConfig.GPU_IDS)
        self.modularity_vgg = nn.DataParallel(Modularity(self.kernel_size, sigma=0.02), ParallelConfig.GPU_IDS)

    def down_sample_to(self, src: Tensor, target: Tensor) -> Tensor:
        down_src = src
        while down_src.shape[-1] != target.shape[-1]:
            down_src = self.downsample(down_src)

        return down_src

    def forward(self, images: Tensor, segmentation: Tensor) -> Loss:

        fich_1 = self.vgg(images)
        fich_1 = nn.BatchNorm2d(fich_1.shape[1]).to(fich_1.device).forward(fich_1).detach()

        # fich_2 = self.vgg_2(images)
        # fich_2 = nn.BatchNorm2d(fich_2.shape[1]).to(fich_2.device).forward(fich_2).detach()

        images = nn.BatchNorm2d(images.shape[1]).to(images.device).forward(images).detach()

        down_segm_1 = self.down_sample_to(segmentation, fich_1)
        # down_segm_2 = self.down_sample_to(down_segm_1, fich_2)

        norm = 2 * ParallelConfig.GPU_IDS.__len__()

        return Loss(
            self.modularity_vgg(fich_1, down_segm_1).sum()
            # + self.modularity_vgg(fich_2, down_segm_2).sum()
            + self.modularity_img(images, segmentation).sum()
        ) / norm

