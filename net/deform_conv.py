import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmcv.ops import modulated_deform_conv2d


class DCN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True, extra_offset_mask=True):
        super(DCN_layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        self.extra_offset_mask = extra_offset_mask
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels * 2,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.init_offset()
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_feat, inter):
        feat_degradation = torch.cat([input_feat, inter], dim=1)

        out = self.conv_offset_mask(feat_degradation)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(input_feat.contiguous(), offset, mask, self.weight, self.bias, self.stride,
                                       self.padding, self.dilation, self.groups, self.deformable_groups)
