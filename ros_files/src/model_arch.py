#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from imu_files.msg import collect_data
from takktile_ros.msg import Touch
import rospy
from std_msgs.msg import Float32
import time
import glob
import rospy
import torch
import torch.nn.functional as F
import copy


#######################################################
class ConvNet(torch.nn.Module):
  def __init__(self, H_in, W_in, D_out, ch_out, H, kernel_w=3, stride_size=1, drop=0.0):
    """
    In the constructor we instantiate Convlolutional and one Linear modules and assign them as member variables.
    """
    super(ConvNet, self).__init__()
    self.conv = torch.nn.Sequential(
                  torch.nn.Conv2d(1, ch_out, (H_in, kernel_w), stride=stride_size),
                  torch.nn.ReLU(),
                  torch.nn.Conv2d(ch_out, ch_out, (H_in, kernel_w), stride=stride_size),
                  torch.nn.ReLU(),
                  torch.nn.Conv2d(ch_out, ch_out, (H_in, kernel_w), stride=stride_size),
                  torch.nn.ReLU()
                )
    W_out = ((W_in - kernel_w) // stride_size + 1)
    W_out = ((W_out - kernel_w) // stride_size + 1)
    W_out = ch_out * ((W_out - kernel_w) // stride_size + 1)
    #pdb.set_trace()
    #self.linear1 = torch.nn.Linear(W_out, D_out)
    self.linear1 = torch.nn.Linear(W_out, H)
    self.linear2 = torch.nn.Linear(H, D_out)
    self.drop1   = torch.nn.Dropout(p=drop)
    # self.sigmoid = torch.nn.Sigmoid()
    self.softmax = torch.nn.Softmax(dim = 1)
  def forward(self, x):
    """
    In the forward function we accept a Tensor of input data and we must return
    a Tensor of output data. We can use Modules defined in the constructor as
    well as arbitrary operators on Tensors.
    """
    (b,w) = x.shape
    h_cnn = self.drop1(self.conv(x.view(b,1,1,w)))
    b, c, h, w = h_cnn.size()
    #pdb.set_trace()
    h_relu = self.drop1(self.linear1(h_cnn.view(b,-1))).clamp(min=0)
    y_pred = self.softmax(self.linear2(h_relu))
    return y_pred

#######################################################
#######################################################

#######################################################
class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out, drop):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as member variables.
    """
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear1_1 = torch.nn.Linear(H, H)
    self.linear1_2 = torch.nn.Linear(H, H)
    self.linear2 = torch.nn.Linear(H, D_out)
    self.drop1   = torch.nn.Dropout(p=drop)
    self.softmax = torch.nn.Softmax(dim = 1)
    # self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):                                
    """
    In the forward function we accept a Tensor of input data and we must return
    a Tensor of output data. We can use Modules defined in the constructor as
    well as arbitrary operators on Tensors.
    """
    h_relu = self.drop1(self.linear1(x)).clamp(min=0)
    h_relu = self.drop1(self.linear1_1(h_relu)).clamp(min=0)
    h_relu = self.drop1(self.linear1_2(h_relu)).clamp(min=0)
    h_relu = self.linear2(h_relu)
    y_pred = self.Softmax(h_relu)
    return y_pred

#######################################################
