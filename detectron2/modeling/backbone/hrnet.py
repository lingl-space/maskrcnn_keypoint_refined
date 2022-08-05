# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "BasicBlock",
    "Bottleneck",
    "HighResolutionModule",
    "HRNet",
    "build_hrnet_backbone",
]

class BasicBlock(CNNBlockBase):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__(inplanes, planes, stride)
        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = Conv2d(inplanes, planes, kernel_size=3,
                            stride=stride, padding=1, bias=False)
        self.bn1 = get_norm('FrozenBN', planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(planes, planes, kernel_size=3,
                            stride=stride, padding=1, bias=False)
        self.bn2 = get_norm('FrozenBN', planes)

        if self.inplanes != self.planes * self.expansion:
            self.downsample = nn.Sequential(
                Conv2d(self.inplanes, self.planes * self.expansion,
                       kernel_size=1, stride=stride, bias=False),
                get_norm('FrozenBN', self.planes * self.expansion)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.inplanes != self.planes * self.expansion:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(CNNBlockBase):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__(inplanes, planes, stride)
        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = get_norm('FrozenBN', planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm('FrozenBN', planes)
        self.conv3 = Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = get_norm('FrozenBN', planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if self.inplanes != self.planes * self.expansion:
            self.downsample = nn.Sequential(
                Conv2d(self.inplanes, self.planes * self.expansion,
                       kernel_size=1, stride=stride, bias=False),
                get_norm('FrozenBN', self.planes * self.expansion),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.inplanes != self.planes * self.expansion:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        get_norm('FrozenBN', num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                       3, 2, 1, bias=False),
                                get_norm('FrozenBN', num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                       3, 2, 1, bias=False),
                                get_norm('FrozenBN', num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class HRNet(Backbone):
    def __init__(self, cfg):
        super(HRNet, self).__init__()

        blocks_dict = {
            'BasicBlockWithFixedBatchNorm': BasicBlock,
            'BottleneckWithFixedBatchNorm': Bottleneck
        }

        self.blocks_dict = blocks_dict
        self.inplanes = 64

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = get_norm('FrozenBN', 64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = get_norm('FrozenBN', 64)
        self.relu = nn.ReLU(inplace=True)

        # stage1
        num_channels = 64
        block = blocks_dict['BottleneckWithFixedBatchNorm']
        num_blocks = 4
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # self.stage2_cfg = cfg.MODEL.HRNET.STAGE2
        num_channels = [32, 64]
        block = blocks_dict['BasicBlockWithFixedBatchNorm']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            1, 2, [4, 4], [32, 64], blocks_dict['BasicBlockWithFixedBatchNorm'], 'SUM',
            num_channels)

        # self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = [32, 64, 128]
        block = blocks_dict['BasicBlockWithFixedBatchNorm']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            4, 3, [4, 4, 4], [32, 64, 128], blocks_dict['BasicBlockWithFixedBatchNorm'], 'SUM',
            num_channels)

        # self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = [32, 64, 128, 256]
        block = blocks_dict['BasicBlockWithFixedBatchNorm']
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            3, 4, [4, 4, 4, 4], [32, 64, 128, 256], blocks_dict['BasicBlockWithFixedBatchNorm'], 'SUM',
            num_channels, multi_scale_output=True)

        # self.final_layer = nn.Conv2d(
        #     in_channels=pre_stage_channels[0],
        #     out_channels=17,
        #     kernel_size=1,
        #     stride=1,
        #     padding=1 if 1 == 3 else 0
        # )
        #
        # self.pretrained_layers = [
        #     'conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1',
        #     'stage2', 'transition2', 'stage3', 'transition3', 'stage4',
        # ]

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        Conv2d(num_channels_pre_layer[i],
                               num_channels_cur_layer[i],
                                3, 1, 1, bias=False),
                        get_norm('FrozenBN', num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        get_norm('FrozenBN', outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        # downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(
        #             self.inplanes, planes * block.expansion,
        #             kernel_size=1, stride=stride, bias=False
        #         ),
        #         nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
        #     )

        layers = []
        layers.append(block(inplanes, planes, stride))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, num_modules, num_branches, num_blocks, num_channels,
                    block, fuse_method,
                    num_inchannels,
                    multi_scale_output=True):

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return tuple(y_list)

@BACKBONE_REGISTRY.register()
def build_hrnet_backbone(cfg, input_shape):
    model = HRNet(cfg)
    return model
