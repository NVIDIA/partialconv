###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#   
# Copyright (c) 2017, Soumith Chintala. All rights reserved.
###############################################################################
'''
Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
Introduced partial convolutoins based padding for convolutional layers
'''

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from .partialconv2d import *

__all__ = [
    'PDVGG', 'pdvgg11', 'pdvgg11_bn', 'pdvgg13', 'pdvgg13_bn', 'pdvgg16', 'pdvgg16_bn',
    'pdvgg19_bn', 'pdvgg19',
]
# __all__ = [
#     'PDVGG', 'pdvgg16_bn', 'pdvgg19_bn',
# ]


model_urls = {
    'pdvgg16_bn': '',
    'pdvgg19_bn': '',
    'pdvgg16': '',
    'pdvgg19': '', 
    'pdvgg11': '',
    'pdvgg13': '',
    'pdvgg11_bn': '',
    'pdvgg13_bn': '',
    # 'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    # 'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    # 'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    # 'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    # 'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    # 'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    # 'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    # 'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class PDVGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(PDVGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, PartialConv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0, 0.01)
                nn.init.constant(m.bias, 0)


def make_layers_pd(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = PartialConv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)




cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def pdvgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = PDVGG(make_layers_pd(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdvgg11']))
    return model


def pdvgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = PDVGG(make_layers_pd(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdvgg11_bn']))
    return model


def pdvgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = PDVGG(make_layers_pd(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdvgg13']))
    return model


def pdvgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = PDVGG(make_layers_pd(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdvgg13_bn']))
    return model


def pdvgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = PDVGG(make_layers_pd(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdvgg16']))
    return model


def pdvgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = PDVGG(make_layers_pd(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdvgg19']))
    return model


def pdvgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = PDVGG(make_layers_pd(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdvgg16_bn']))
    return model


def pdvgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = PDVGG(make_layers_pd(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdvgg19_bn']))
    return model