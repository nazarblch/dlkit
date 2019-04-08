import torch


class GradientPenalty:

    def __init__(self, weight, discriminator):
        self.weight = weight
        self.discriminator = discriminator

    def __call__(self, real, fake):
        batch_size = real.size(0)

        alpha = torch.rand(batch_size, *((1,) * (real.ndimension() - 1))).cuda()

        hat_x = alpha * real.data + (1 - alpha) * fake.data
        hat_x.requires_grad = True

        hat_y = self.discriminator(hat_x)

        gradients = torch.autograd.grad(outputs=hat_y,
                                        inputs=hat_x,
                                        grad_outputs=torch.ones(hat_y.size()).cuda(),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty_value = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self.weight * gradient_penalty_value
