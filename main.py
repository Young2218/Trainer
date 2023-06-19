import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from model.blocks.attentionBlock import AttentionBlock, ChannelAttention, SpatialAttention
from model.blocks.doubleConvBlock import DoubleConvBlock
from model.blocks.upConvBlock import UpConvBlock
from model.tecdv2 import TECDv2

from _Project.TECDv2.train_tecd_v2 import *

if __name__ == "__main__":
    train_TECDv2()