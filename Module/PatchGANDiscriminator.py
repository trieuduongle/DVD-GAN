import torch
import torch.nn as nn
from Module.Modules import BaseModule

from torch.autograd import Variable

import numpy as np

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch

# class SNTemporalPatchGANDiscriminator(BaseModule):
#     def __init__(
#         self, nc_in, nf=64, norm='SN', use_sigmoid=True, use_bias=True, conv_type='vanilla',
#         conv_by='3d'
#     ):
#         super().__init__(conv_type)
#         use_bias = use_bias
#         self.use_sigmoid = use_sigmoid

#         ######################
#         # Convolution layers #
#         ######################
#         self.conv1 = self.ConvBlock(
#             nc_in, nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
#             padding=1, bias=use_bias, norm=norm, conv_by=conv_by
#         )
#         self.conv2 = self.ConvBlock(
#             nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
#             padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
#         )
#         self.conv3 = self.ConvBlock(
#             nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
#             padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
#         )
#         self.conv4 = self.ConvBlock(
#             nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
#             padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
#         )
#         self.conv5 = self.ConvBlock(
#             nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
#             padding=(1, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
#         )
#         self.conv6 = self.ConvBlock(
#             nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
#             padding=(1, 2, 2), bias=use_bias, norm=None, activation=None,
#             conv_by=conv_by
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, xs, transpose = True):
#         # B, L, C, H, W = xs.shape
#         # B, C, L, H, W = xs_t.shape
#         xs_t = xs
#         if transpose:
#             xs_t = torch.transpose(xs, 1, 2)
#         c1 = self.conv1(xs_t)
#         c2 = self.conv2(c1)
#         c3 = self.conv3(c2)
#         c4 = self.conv4(c3)
#         c5 = self.conv5(c4)
#         c6 = self.conv6(c5)
#         if self.use_sigmoid:
#             c6 = torch.sigmoid(c6)
#         out = torch.transpose(c6, 1, 2)
#         return out

class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * Variable(T.FloatTensor(x.size()).normal_(), requires_grad=False)
        return x

class SNTemporalPatchGANDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(SNTemporalPatchGANDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 8, n_output_neurons, 4, 1, 0, bias=False),
        )

    def forward(self, input, transpose = True):
        xs_t = input
        if transpose:
            xs_t = torch.transpose(xs_t, 1, 2)
        h = self.main(xs_t).squeeze()

        return h
