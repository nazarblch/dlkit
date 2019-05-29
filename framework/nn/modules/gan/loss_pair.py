from framework.Loss import Loss


class GANLossPair:
    def __init__(self, generator_loss: Loss, discriminator_loss: Loss):
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    def add_generator_loss(self, loss: Loss):
        self.generator_loss += loss

    def add_discriminator_loss(self, loss: Loss):
        self.discriminator_loss += loss
