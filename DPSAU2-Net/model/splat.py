"""Split-Attention"""
"""
Reference:

- Zhang, Hang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun et al. "Resnest: Split-attention networks." arXiv preprint arXiv:2004.08955 (2020)
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair

__all__ = ['SplAtConv2d']



class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, groups=1, bias=True, radix=2, reduction_factor=4, **kwargs):
        super(SplAtConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.conv = Conv2d(in_channels, channels * radix, kernel_size, padding=kernel_size // 2, groups=groups * radix,
                           bias=bias,
                           **kwargs)
        self.bn0 = nn.BatchNorm2d(channels * radix)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)
        self.conv2 = Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=groups * radix, bias=bias,
                            **kwargs)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        x1, x2  = torch.split(x, rchannel // self.radix, dim=1)
        x3 = x2 + x1
        splited = (x1, x2)
        gap = x3
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        attens = torch.split(atten, rchannel // self.radix, dim=1)

        out = sum([att * split for (att, split) in zip(attens, splited)])
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x
