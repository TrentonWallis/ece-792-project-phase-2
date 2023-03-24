from helper_functions import *
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as func


class CNNNet(nn.Module):
    def __init__(self) -> None:
        super(CNNNet, self).__init__()
        self.conv1 =  nn.ConvTranspose2d(3, 40, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(40, 3, 4, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(40) 
        self.batchnorm2 = nn.BatchNorm2d(3)

# https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11#:~:text=Transposed%20convolutions%20are%20standard%20convolutions,in%20a%20standard%20convolution%20operation.
    def forward(self, x):
        x = self.batchnorm1(func.relu(self.conv1(x)))
        x = self.batchnorm2(func.relu(self.conv2(x)))
        return func.log_softmax(x)

class Dense_Connection_Model(nn.Module):
    # https://www.analyticsvidhya.com/blog/2021/05/deep-learning-for-image-super-resolution/#:~:text=Super%2DResolution%20Generative%20Adversarial%20Network,match%20the%20true%20training%20data.
    # option (d) from above
    # arXiv:1501.00092v3 [cs.CV] 31 Jul 2015 -> simple (used this one)
    # arXiv:1707.02921v1 [cs.CV] 10 Jul 2017 -> complex


    def __init__(self) -> None:
        super(Dense_Connection_Model, self).__init__()
        self.up_2x = nn.Upsample(scale_factor = 4, mode='bicubic')

        self.cnn1 = nn.Conv2d(3, 64, 9, padding = 'same')
        self.cnn2 = nn.Conv2d(64, 32, 1, padding = 'same')
        self.output = nn.Conv2d(32, 3, 5, padding = 'same')
        
    def forward(self, x):
        x = self.up_2x(x)
        x = func.relu(self.cnn1(x))
        x = func.relu(self.cnn2(x))
        return self.output(x)
    
class Residual_Model(nn.Module):
    def __init__(self) -> None:
        super(Residual_Model, self).__init__()
        self.up_2x = nn.Upsample(scale_factor = 2, mode='nearest')

        # do a convtranspose2d????
        self.conv1 = nn.Conv2d(3, 16, 2, padding = 'same')
        self.batchnorm1 = nn.BatchNorm2d(16)

        self.upconv1 =  nn.ConvTranspose2d(16, 64, 2, stride=2, padding=0)
        # concatenate upsampled
        self.batchnorm2 = nn.BatchNorm2d(64 + 3)

        self.conv2 = nn.Conv2d(64 + 3, 128, 2, padding = 'same')
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.upconv2 = nn.ConvTranspose2d(128, 32, 2, stride=2, padding=0)
        # concatenate upsampled
        self.batchnorm4 = nn.BatchNorm2d(32 + 3)

        self.conv3 = nn.Conv2d(32+3, 3, 2, padding = 'same')

        # self.batchnorm1 = nn.BatchNorm2d(40) 
        # self.batchnor
        m2 = nn.BatchNorm2d(3)

# https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11#:~:text=Transposed%20convolutions%20are%20standard%20convolutions,in%20a%20standard%20convolution%20operation.
    def forward(self, x):
        # convolve 
        input = x.detach().clone()
        x = self.batchnorm1(func.relu(self.conv1(x)))

        # upconv and concatenate upsampled 
        # print(torch.cat((self.up_2x(input), self.upconv1(x)), dim=1).shape)
        x = torch.cat((self.up_2x(input), self.upconv1(x)), dim=1)
        x = self.batchnorm2(func.relu(x))

        # convolve
        x = self.batchnorm3(func.relu(self.conv2(x)))

        # up conv and concatenate upsampled
        x = torch.cat((self.up_2x(self.up_2x(input)), self.upconv2(x)), dim=1)
        # x = self.batchnorm4(func.relu(x))
        return self.conv3(x)
    
# build residual block
class SR_Residual_Block(nn.Module):
  def __init__(self, input_Ls, output_Ls) -> None:
    super().__init__()
    # RESIDUAL blocks:
      # B1: 
        # input -> conv -> BN -> PReLU -> conv -> BN -> element sum
        #  |                                             ^
        #   ---------------------------------------------|
    self.block = nn.Sequential(
        nn.Conv2d(input_Ls, output_Ls, 3, stride = 1, padding = 'same'), 
        nn.BatchNorm2d(output_Ls), 
        nn.PReLU(), 
        nn.Conv2d(output_Ls, output_Ls, 3, stride = 1, padding = 'same'), 
        nn.BatchNorm2d(output_Ls), 
    )

  def forward(self, x):
      x = torch.add(self.block(x), x)
      return x

# build PixelShuffle block
class SR_PixelShuffle_Block(nn.Module):
  def __init__(self, input_Ls, output_Ls) -> None:
    super().__init__()
    self.conv = nn.Conv2d(input_Ls, output_Ls, 3, stride = 1, padding = 'same')
    self.prelu = nn.PReLU()

  def forward(self, x):
      return self.prelu(nn.functional.pixel_shuffle(self.conv(x),2))

class SR_Generator(nn.Module):
  #https://arxiv.org/pdf/1609.04802.pdf
    def __init__(self) -> None:
      super(SR_Generator, self).__init__()
      self.conv_in = nn.Conv2d(3, 64, 3, stride = 1, padding = 'same')
      self.prelu_in = nn.PReLU()
      self.conv_middle = nn.Conv2d(64, 64, 3, stride = 1, padding = 'same')
      self.bn = nn.BatchNorm2d(64)
      self.conv_out = nn.Conv2d(64, 3, 9, stride = 1, padding = 'same')


      
      self.B1 = SR_Residual_Block(64, 64)
      self.B2 = SR_Residual_Block(64, 64)
      self.B3 = SR_Residual_Block(64, 64)
      self.B4 = SR_Residual_Block(64, 64)
      self.B5 = SR_Residual_Block(64, 64)
      
      self.pixel1 = SR_PixelShuffle_Block(64,256)
      self.pixel2 = SR_PixelShuffle_Block(64,256)

    def forward(self, x):
      x = self.prelu_in(self.conv_in(x))

      # skip connection
      x_copy = x.detach().clone()
      
      # residual layers
      x = self.B1(x)
      x = self.B2(x) 
      x = self.B3(x) 
      x = self.B4(x) 
      x = self.B5(x)

      x = torch.add(self.conv_middle(x), x_copy)
      x = self.pixel1(x)
      x = self.pixel2(x)

      return self.conv_out(x)


# build residual block
class Ex_SR_Residual_Block(nn.Module):
  def __init__(self, input_Ls, output_Ls) -> None:
    super().__init__()
    # RESIDUAL blocks:
      # B1: 
        # input -> conv -> BN -> PReLU -> conv -> BN -> element sum
        #  |                                             ^
        #   ---------------------------------------------|
    self.block = nn.Sequential(
        nn.Conv2d(input_Ls, output_Ls, 3, stride = 1, padding = 'same'), 
        nn.BatchNorm2d(output_Ls), 
        nn.PReLU(), 
        nn.Conv2d(output_Ls, output_Ls, 3, stride = 1, padding = 'same'), 
        nn.BatchNorm2d(output_Ls),
        nn.Conv2d(output_Ls, output_Ls, 1, stride = 1, padding = 'same'),
        nn.PReLU()
    )

  def forward(self, x):
      x = torch.add(self.block(x), x)
      return x


class Ex_SR_Generator(nn.Module):
  #https://arxiv.org/pdf/1609.04802.pdf
    def __init__(self) -> None:
      super(Ex_SR_Generator, self).__init__()
      self.UP = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

      # first upscale + residual blocks
      self.UP_1 = nn.ConvTranspose2d(3, 32, kernel_size = 2, stride = 2)
      self.B1_a = Ex_SR_Residual_Block(32, 32)
      self.B2_a = Ex_SR_Residual_Block(32, 32)
      self.B3_a = Ex_SR_Residual_Block(32, 32)
      self.B4_a = Ex_SR_Residual_Block(32, 32)

      # first upscale + residual blocks
      self.UP_2 = nn.ConvTranspose2d(32, 64, kernel_size = 2, stride = 2)
      self.B1_b = Ex_SR_Residual_Block(64, 64)
      self.B2_b = Ex_SR_Residual_Block(64, 64)
      self.B3_b = Ex_SR_Residual_Block(64, 64)
      self.B4_b = Ex_SR_Residual_Block(64, 64)

      # aggregation section
      # B3_output -> conv -> BN -> add UP_1
      self.agg = nn.Sequential(
          nn.Conv2d(64, 3, kernel_size = 3, stride = 1, padding = 'same'), 
          nn.PReLU()
      )
      
      self.output = nn.Sequential(
          nn.Conv2d(3, 3, 1, stride = 1, padding = 'same'),
        )
      self.sigmoid = nn.Sigmoid()
    def forward(self, x):
      # copy upsampled input
      x_copy = self.UP(x).detach().clone()
      
      # increase dims
      x = self.UP_1(x)
      # pass through 1st set of residual blocks
      x = self.B1_a(x)
      x = self.B2_a(x)
      x = self.B3_a(x)
      x = self.B4_a(x)

      # increase dims
      x = self.UP_2(x)
      # pass through set of residual blocks
      x = self.B1_b(x)
      x = self.B2_b(x)
      x = self.B3_b(x)
      x = self.B4_b(x)

      x = self.agg(x)
      x = torch.add(x, x_copy)

      return self.sigmoid(self.output(x))
    

