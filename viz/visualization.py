import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def decode_segmap(temp):
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [52, 11, 152],
        [152, 51, 52],
    ]

    label_colours = dict(zip(range(33), colors))

    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 33):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def segmap(labels, classes, void=(255, 255, 255)):
    labels = np.asarray(labels)
    img = np.empty(labels.shape + (3,))
    img[:] = void
    for i, _, color in classes:
        img[labels == i] = color
    img /= 255
    return img


def mask_to_rgb(mask, colors):
    if mask.dim() == 4:
        size = mask.shape[-2:]
    elif mask.dim() == 5:
        size = mask.shape[-3:]
    else:
        raise ValueError('4D or 5D input tensor is expected')
    last_dim_index = len(size) + 1
    return (colors[mask.data.max(1)[1].flatten()]
            .view(*((mask.size(0),) + size + (3,)))
            .permute(0, last_dim_index, *range(1, last_dim_index)))


def get_color_table(dataset, device='cpu', dtype=torch.float32):
    colors = [color for _, _, color in dataset.classes]
    return torch.tensor(colors, device=device, dtype=dtype) / 255

def show_images(imlist, rows, cols):
    plt.figure(figsize=(rows, cols))
    plt.imshow(np.transpose(
        vutils.make_grid(imlist[:rows * cols], normalize=True, nrow=rows),
        (1, 2, 0)
    ))
    plt.show()


def show_segmentation(segmlist):
    segm_imgs = segmlist[:4].max(1)[1].cpu().numpy()

    f, axarr = plt.subplots(2, 2)

    axarr[0][0].imshow(decode_segmap(segm_imgs[0]))
    axarr[0][1].imshow(decode_segmap(segm_imgs[1]))
    axarr[1][0].imshow(decode_segmap(segm_imgs[2]))
    axarr[1][1].imshow(decode_segmap(segm_imgs[3]))

    plt.show()
