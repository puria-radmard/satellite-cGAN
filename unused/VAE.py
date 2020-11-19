import numpy as np
import torch
from torch import nn

from utils import *


class VAE32DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):

        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        ]

        self.downblock = nn.Sequential(*layers)

    def forward(self, X):
        X = X.float()
        return self.downblock(X)


class VAE32UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()

        # Channels are doubled by upconv
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        ]
        self.dbconv = nn.Sequential(*layers)

    def forward(self, X):

        return self.dbconv(X)


class SamplingLayer(nn.Module):
    def __init__(self, latent_dim):

        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, X):
        "X comes in size (batch_size, latent_dim)"

        t_mean = X[:, : int(self.latent_dim / 2)]
        t_log_var = X[:, int(self.latent_dim / 2) :]

        return torch.normal(t_mean, torch.exp(0.5 * t_log_var))


class VariationalAutoencoder32(nn.Module):
    def __init__(self, latent_dim, dropout):

        super(VariationalAutoencoder32, self).__init__()

        self.latent_dim = latent_dim

        layers = [
            VAE32DownBlock(3, 64, dropout),
            VAE32DownBlock(64, 128, dropout),
            VAE32DownBlock(128, 512, dropout),
            LambdaLayer(lambd=lambda X: X.view(X.shape[0], -1)),
            nn.Linear(4608, latent_dim),
            SamplingLayer(latent_dim),
            LambdaLayer(lambd=lambda X: X.view(X.shape[0], int(latent_dim / 32), 4, 4)),
            VAE32UpBlock(int(latent_dim / 32), 128, dropout),
            VAE32UpBlock(128, 64, dropout),
            VAE32UpBlock(64, 3, dropout),
        ]

        self.model = construct_debug_model(layers, False)

    def forward(self, X):

        return self.model(X)


class NDVIOptimiser(nn.Module):
    def __init__(self):

        super(NDVIOptimiser, self).__init__()

        layers = None
