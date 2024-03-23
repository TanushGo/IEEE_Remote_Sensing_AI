"""
This code is adapted from the U-Net paper. See details in the paper:
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. 
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn.functional import relu, pad
from collections import deque

class DoubleConvHelper(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Module that implements 
            - a convolution
            - a batch norm
            - relu
            - another convolution
            - another batch norm
        
        Input:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            mid_channels (int): number of channels to use in the intermediate layer    
        """
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels #checking to see if midchannels is none
        
        #constructing model with different layers and then creating the sequential object
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the layers of the helper block"""
        return self.model.forward(x)


class Encoder(nn.Module):
    """ Downscale using the maxpool then call double conv helper. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #aCreating encoder with Doubleconvhelper and max pooling in a sequential
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvHelper(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Decoder(nn.Module):
    """ Upscale using ConvTranspose2d then call double conv helper. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #creating upscaling and double conv helper for decoders
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvHelper(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """ 
        1) x1 is passed through either the upsample or the convtranspose2d
        2) The difference between x1 and x2 is calculated to account for differences in padding
        3) x1 is padded (or not padded) accordingly
        4) x2 represents the skip connection
        5) Concatenate x1 and x2 together with torch.cat
        6) Pass the concatenated tensor through a doubleconvhelper
        7) Return output
        """
        # step 1: replace x1 with the upsampled version of x1
        x1 = self.up(x1)
        # input is Channel Height Width, step 2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # step 3
        x1 = pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        # step 4 & 5: Concatenate x1 and x2
        x = torch.cat([x2, x1], dim=1)
        

        # step 6: Pass the concatenated tensor through a doubleconvhelper
        
        # step 7: return output
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2,
                 embedding_size: int = 64, scale_factor: int = 50, **kwargs):
        """
        Implements a unet, a network where the input is downscaled
        down to a lower resolution with a higher amount of channels,
        but the residual images between encoders are saved
        to be concatednated to later stages, creatin the
        nominal "U" shape.

        In order to do this, we will need n_encoders-1 encoders. 
        The first layer will be a doubleconvhelper that
        projects the in_channels image to an embedding_size
        image of the same size.

        After that, n_encoders-1 encoders are used which halve
        the size of the image, but double the amount of channels
        available to them (i.e, the first layer is 
        embedding_size -> 2*embedding size, the second layer is
        2*embedding_size -> 4*embedding_size, etc)

        The decoders then upscale the image and halve the amount of
        embedding layers, i.e., they go from 4*embedding_size->2*embedding_size.

        We then have a maxpool2d that scales down the output to by scale_factor,
        as the input for this architecture must be the same size as the output,
        but our input images are 800x800 and our output images are 16x16.

        Input:
            in_channels: number of input channels of the image
            of shape (batch, in_channels, width, height)
            out_channels: number of output channels of prediction,
            prediction is shape (batch, out_channels, width//scale_factor, height//scale_factor)
            n_encoders: number of encoders to use in the network (implementing this parameter is
            optional, but it is a good idea to have it as a parameter in case you want to experiment,
            if you do not implement it, you can assume n_encoders=2)
            embedding_size: number of channels to use in the first layer
            scale_factor: number of input pixels that map to 1 output pixel,
            for example, if the input is 800x800 and the output is 16x6
            then the scale factor is 800/16 = 50.
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        #Creating inital setp for unet
        self.inc = (DoubleConvHelper(in_channels, embedding_size))
        self.encoders = []
        size = embedding_size
        #Creating all the encoders for first half of unet
        for i in range(n_encoders):
            self.encoders.append(Encoder(size,size*2))
            size=size*2

        self.decoders = []
        #Creating all the decoders for second half of unet
        for i in range(n_encoders):
            if(i == n_encoders-1):
                self.decoders.append(Decoder(size,out_channels))
                break
            self.decoders.append(Decoder(size,size//2))
            size=size//2

        #Pooling the output 
        self.outc =nn.MaxPool2d(kernel_size=(scale_factor,scale_factor))


    def forward(self, x):
        """
            The image is passed through the encoder layers,
            making sure to save the residuals in a list.

            Following this, the residuals are passed to the
            decoder in reverse, excluding the last residual
            (as this is used as the input to the first decoder).

            The ith decoder should have an input of shape
            (batch, some_embedding_size, some_width, some_height)
            as the input image and
            (batch, some_embedding_size//2, 2*some_width, 2*some_height)
            as the residual.
        """
        #Doing the first step in the process
        x = self.inc(x)
        residuals = [x]
        #using all the encoders and creating the residuals in a lis
        for i, enc in enumerate(self.encoders):
            residuals.append(enc(residuals[-1]))
        
        #getting encoder output
        x = residuals[-1]
        #using the decoders and combining output with each layers residual
        for i,(dec,res) in enumerate(zip(self.decoders,reversed(residuals[:-1]))):
            x = dec(x,res)

        #Getting the pooled output
        x = self.outc(x)
        return x

