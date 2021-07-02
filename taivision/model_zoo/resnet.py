"""ResNets, implemented in Gluon."""
from __future__ import division

__all__ = ['ResNetV1', 'ResNetV2',
           'BasicBlockV1', 'BasicBlockV2',
           'BottleneckV1', 'BottleneckV2',
           'resnet18_v1', 'resnet34_v1', 'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
           'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
           'get_resnet']

import torch
import torch.nn as nn
from torch.nn import BatchNorm2d as BatchNorm

# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)


# Blocks
class BasicBlockV1(nn.Module):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)

        assert last_gamma==False, 'last_gamma is not yet implemented!'
        assert use_se==False, 'use_se is not yet implemented!'

        body = []
        body.append(_conv3x3(channels, stride, in_channels))
        body.append(norm_layer(channels))
        body.append(nn.ReLU(inplace=True))
        body.append(_conv3x3(channels, 1, channels))
        body.append(norm_layer(channels))
        self.body = nn.Sequential(*body)

        if downsample:
            downsample = []
            downsample.append(nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False))
            downsample.append(norm_layer(channels))
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = None
        
    def forward(self, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual=self.downsample(residual)

        x = nn.ReLU(inplace=True)(x+residual)

        return x


class BottleneckV1(nn.Module):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)

        assert last_gamma==False, 'last_gamma is not yet implemented!'
        assert use_se==False, 'use_se is not yet implemented!'

        body = []
        body.append(nn.Conv2d(in_channels, channels//4, kernel_size=1, stride=stride))
        body.append(norm_layer(channels//4))
        body.append(nn.ReLU(inplace=True))
        body.append(_conv3x3(channels//4, 1, channels//4))
        body.append(norm_layer(channels//4))
        body.append(nn.ReLU(inplace=True))
        body.append(nn.Conv2d(channels//4, channels, kernel_size=1, stride=1))
        self.body = nn.Sequential(*body)

        if downsample:
            downsample = []
            downsample.append(nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False))
            downsample.append(norm_layer(channels))
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        x = self.body(x)
        if self.downsample:
            residual=self.downsample(residual)

        x = nn.ReLU(inplace=True)(x+residual)

        return x


class BasicBlockV2(nn.Module):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)

        assert last_gamma==False, 'last_gamma is not yet implemented!'
        assert use_se==False, 'use_se is not yet implemented!'

        self.bn1 = norm_layer(in_channels)
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = norm_layer(channels)
        self.conv2 = _conv3x3(channels, 1, channels)

        if downsample:
            self.downsample = nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        if self.downsample:
            residual = self.downsample(residual)
        x = self.conv1(x)

        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)

        return x+residual

class BottleneckV2(nn.Module):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)

        assert last_gamma==False, 'last_gamma is not yet implemented!'
        assert use_se==False, 'use_se is not yet implemented!'

        self.bn1 = norm_layer(in_channels)
        self.conv1 = nn.Conv2d(in_channels, channels//4, kernel_size=1, stride=1, bias=False)
        self.bn2 = norm_layer(channels//4)
        self.conv2 = _conv3x3(channels//4, stride, channels//4)
        self.bn3 = norm_layer(channels//4)
        self.conv3 = nn.Conv2d(channels//4, channels, kernel_size=1, stride=1, bias=False)

        if downsample:
            self.downsample = nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
    
    def forward(self, x):
        residual = x

        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        if self.downsample:
            residual = self.downsample(residual)
        x = self.conv1(x)

        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv3(x)

        return x+residual


# Nets
class ResNetV1(nn.Module):
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1

        assert thumbnail==False, 'thumbnail is not yet implemented!'
        assert last_gamma==False, 'last_gamma is not yet implemented!'
        assert use_se==False, 'use_se is not yet implemented!'

        features = []
        features.append(nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False))
        features.append(norm_layer(channels[0]))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(3, 2, 1))

        for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                features.append(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i],
                                                   last_gamma=last_gamma, use_se=use_se,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))
        features.append(nn.Flatten(1))
        
        self.features = nn.Sequential(*features)

        self.output = nn.Linear(channels[-1], classes)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None):
        layer = []
        layer.append(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, 
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        for _ in range(layers-1):
            layer.append(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, 
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)

        return x


class ResNetV2(nn.Module):
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1

        assert thumbnail==False, 'thumbnail is not yet implemented!'
        assert last_gamma==False, 'last_gamma is not yet implemented!'
        assert use_se==False, 'use_se is not yet implemented!'

        features = []
        features.append(norm_layer(3))

        features.append(nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False))
        features.append(norm_layer(channels[0]))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(3, 2, 1))

        in_channels=channels[0]
        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            features.append(self._make_layer(block, num_layer, channels[i+1],
                                                stride, i+1, in_channels=in_channels,
                                                last_gamma=last_gamma, use_se=use_se,
                                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            in_channels = channels[i+1]
        features.append(norm_layer(in_channels))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.AdaptiveAvgPool2d(1))
        features.append(nn.Flatten(1))

        self.features = nn.Sequential(*features)

        self.output = nn.Linear(in_channels, classes)
    
    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None):
        layer = []
        layer.append(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            last_gamma=last_gamma, use_se=use_se, 
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        for _ in range(layers-1):
            layer.append(block(channels, 1, False, in_channels=channels,
                                last_gamma=last_gamma, use_se=use_se, 
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1, ResNetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]


# Constructor
def get_resnet(version, num_layers, pretrained=False, ctx=False,
               root='~/.mxnet/models', use_se=False, **kwargs):
    
    assert pretrained==False, 'pretrained is not yet implemented!'
    assert ctx==False, 'ctx is not yet implemented!'
    assert use_se==False, 'use_se is not yet implemented!'

    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert 1 <= version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version
    
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, use_se=use_se, **kwargs)
    
    return net
    
def resnet18_v1(**kwargs):
    return get_resnet(1, 18, use_se=False, **kwargs)

def resnet34_v1(**kwargs):
    return get_resnet(1, 34, use_se=False, **kwargs)

def resnet50_v1(**kwargs):
    return get_resnet(1, 50, use_se=False, **kwargs)

def resnet101_v1(**kwargs):
    return get_resnet(1, 101, use_se=False, **kwargs)

def resnet152_v1(**kwargs):
    return get_resnet(1, 152, use_se=False, **kwargs)

def resnet18_v2(**kwargs):
    return get_resnet(2, 18, use_se=False, **kwargs)

def resnet34_v2(**kwargs):
    return get_resnet(2, 34, use_se=False, **kwargs)

def resnet50_v2(**kwargs):
    return get_resnet(2, 50, use_se=False, **kwargs)

def resnet101_v2(**kwargs):
    return get_resnet(2, 101, use_se=False, **kwargs)

def resnet152_v2(**kwargs):
    return get_resnet(2, 152, use_se=False, **kwargs)















































