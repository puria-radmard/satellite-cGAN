import torch
from torch import nn
from models.unet import UNet
from models.fpn import FPN


class Discriminator(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        """
        TODO: parameterise architecture?
        """
        super(Discriminator, self).__init__()

        self.num_classes = num_classes
        self.dropout = dropout

        convchan1 = 64
        convchan2 = 128
        convchan3 = 256
        convchan4 = 512
        convchan5 = 1024

        layers = [
            # For now we assume all images have 1 channel
            UNetDownBlock(1, convchan1, dropout, max_before=False),
            UNetDownBlock(convchan1, convchan2, dropout),
            UNetDownBlock(convchan2, convchan3, dropout),
            UNetDownBlock(convchan3, convchan4, dropout),
            # TODO: Should I change ReLU for this last one? Don't think so right?
            UNetDownBlock(convchan4, convchan5, dropout),
            # The final size here is [N x 1024 x 16 x 16] as per UNet specifications
            # I have added a mean over the last two dimensions here, but it's not
            # that pretty, there must be a better way
            LambdaLayer(lambd=lambda X: X.mean(-1).mean(-1)),
            # Fully connected -> real probability for each type
            nn.Linear(convchan5, self.num_classes),
            nn.Sigmoid(),
        ]

        self.model = construct_debug_model(layers, False)

    def forward(self, X, reorder=True):
        """
        X comes in as [N, H, W, C]
        """
        if len(np.shape(X)) == 3:
            X = X[np.newaxis, :]
        if reorder:
            X = X.permute(0, 3, 1, 2)
        if np.shape(X)[1] == 4:
            X = X[:, :3, :, :]

        logits = self.model(X)
        return logits


class ConditionalGAN(nn.Module):
    def __init__(
        self,
        classes,
        channels,
        dis_dropout,
        gen_dropout,
        no_discriminator,
        sigmoid_channels,
        generator_class,
        generator_params,
    ):

        # Is this needed?
        super(ConditionalGAN, self).__init__()
        self.classes = classes
        self.channels = channels

        if no_discriminator:
            self.has_discriminator = False

        else:
            self.has_discriminator = True
            self.discriminator = Discriminator(
                num_classes=len(classes), dropout=dis_dropout
            )

        self.generator = generator_class(
            dropout=gen_dropout,
            n_channels=len(channels),
            n_classes=len(classes),
            sigmoid_channels=sigmoid_channels,
            **generator_params
        )
