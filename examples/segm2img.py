# Root directory for dataset
import torch

from framework.data_loader.data2d.datasets import Cityscapes
from framework.nn.modules.gan.image2image.gan_factory import GANFactory

dataroot = "../../celeba"
# Number of workers for dataloader
workers = 12
# Batch size during training
batch_size = 30
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 30

# Create the dataloader
dataset = Cityscapes(dataroot)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


G_losses = []
D_losses = []

netG, netD = GANFactory(image_size, nz, ngf, ndf, nc, device)


print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (imgs, labels) in enumerate(dataloader, 0):

        imgs_cuda = imgs.to(device)
        mask = class_map_to_mask(labels.to(device))

        # segmentation_module.zero_grad()
        # mask_pred = scores2mask(segmentation_module(imgs_cuda))
        latent = torch.FloatTensor(imgs_cuda.size(0), nz)
        latent = latent.normal_().to(device)
        G_loss, D_loss = ganModel.make_train_step(imgs_cuda, latent, mask)
        # segmentation_optimizer.step()

        # ganModel.G.zero_grad()
        # segmentation_module.zero_grad()
        # mask_pred = scores2mask( segmentation_module(imgs_cuda) )
        # fake = ganModel.G(latent, mask_pred)
        # fake_mask_pred = scores2mask( segmentation_module(fake) )
        # cycle_loss = loss_fn(fake_mask_pred, mask_pred.max(1)[1]) + loss_fn(mask_pred, fake_mask_pred.max(1)[1])
        # (0.2 * cycle_loss).backward()
        # ganModel.optimizerG.step()
        # segmentation_optimizer.step()

        G_losses.append(G_loss)
        D_losses.append(D_loss)

        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tD_Loss: %.4f\tG_Loss: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                        sum(D_losses)/len(D_losses), sum(G_losses)/len(G_losses)))

        # Check how the generator is doing by saving G's output on fixed_noise
        if (i % 50 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                imlist = ganModel.G(latent, mask).detach().cpu()
            show_images(imlist, 2, 2)
            show_segmentation(mask)






