# This file contains the models used for both parts of the assignment:
#
#   - CycleGenerator    --> Used in the CycleGAN in Part 2
#   - DCDiscriminator   --> Used in both the vanilla GAN and CycleGAN (Parts 1 and 2)
#
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ methods in the DCGenerator, CycleGenerator, and DCDiscriminator classes.
# Note that the forward passes of these models are provided for you, so the only part you need to
# fill in is __init__.

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_dim, out_dim, kernel_size, stride, padding)
        
    def forward(self, x):
        out = F.relu(self.conv_layer(x))
        out = x + self.conv_layer(out)
        return out

class CycleGenerator_v2(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self):
        super(CycleGenerator_v2, self).__init__()
        
        # 1. Define the encoder part of the generator (that extracts features from the input image)
        #
        conv1 = conv(3, 64, 7, stride=1, padding=3) # (B, 64, 256, 256) + BN
        conv2 = conv(64, 128, 3) # (B, 128, 256, 256) + BN
        conv3 = conv(128, 256, 3) # (B, 256, 128, 128) + BN
        
        resnet4 = ResnetBlock(256,256) # (B, 512, 128, 128) + BN + Residual + ReLU
        resnet5 = ResnetBlock(256,256)
        resnet6 = ResnetBlock(256,256)
        resnet7 = ResnetBlock(256,256)
        resnet8 = ResnetBlock(256,256)
        resnet9 = ResnetBlock(256,256)

        self.Encoder = nn.Sequential(*[conv1, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU(),
                                       resnet4, resnet5, resnet6, resnet7, resnet8, resnet9
                                      ])
        
        # 3. Define the decoder part of the generator (that builds up the output image from features)
        #
        self.deconv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1) # (B, 128, 256, 256)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 3, 3, stride=2 ,padding = 1) # (B, 3, 256, 256)
        self.bn4 = nn.BatchNorm2d(3)
        self.conv_last = conv(3, 3, 7, stride=1, padding=3) # (B, 3, 256, 256) + BN
        
    
    def forward(self, x):
        """Generates an image conditioned on an input image.
            Input
            -----
                x: B x 3 x 256 x 256
            Output
            ------
                out: B x 3 x 256 x 256
        """
        out = self.Encoder(x)
        
        out = F.relu(self.bn2(self.deconv1(out, output_size = (-1,128,128,128))))
        out = F.relu(self.bn4(self.deconv3(out, output_size = (-1,3,256,256))))
        out = F.tanh(self.conv_last(out))
        
        return out


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self):
        super(DCDiscriminator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        self.conv1 = conv(3, 32, 5 , stride = 1 , padding = 2) # (B, 32, 256, 256)
        self.conv2 = conv(32, 64, 4) # (B, 64, 128, 128)
        self.conv3 = conv(64, 128, 4) # (B, 128, 64, 64)
        self.conv4 = conv(128, 256, 4) # (B, 256, 32, 32)
        self.conv5 = conv(256, 512, 4) # (B, 512, 16, 16)
        self.conv6 = conv(512, 1, 16, padding = 0, batch_norm = False) # (B, 512, 1, 1)

    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        
        out = self.conv6(out).squeeze()
        out = F.sigmoid(out)
        return out
