from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import os
import sys
cwd = os.getcwd()
model_cwd = os.path.join(cwd, 'model')
sys.path.append(os.path.abspath(model_cwd))
from meta_modules import MetaBatchNorm2d, MetaConv2d, MetaLinear, MetaModule


model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nobn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = MetaBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = MetaBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.nobn = nobn

    def forward(self, x, source=True):

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, nobn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1,
                               stride=stride, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)

        self.conv2 = MetaConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)

        self.conv3 = MetaConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride
        self.nobn = nobn

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
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(MetaModule):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, nobn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nobn=nobn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet34(pretrained=True):
    """Constructs a ResNet-34 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet101(pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model



class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, lambd=1.0):
    return ReverseLayerF.apply(x, lambd)


class Predictor_deep(MetaModule):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = MetaLinear(inc, 512)
        self.fc2 = MetaLinear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out


class MetaResnet34(MetaModule):
    def __init__(self, num_class=64, inc=512, temp=0.05):
        super(MetaResnet34, self).__init__()
        self.G = resnet34()
        del self.G.fc
        self.F1 = Predictor_deep(num_class=num_class, inc=inc, temp=temp)
        weights_init(self.F1)

    def forward(self, x):
        x = self.G(x)
        x = self.F1(x)
        return x


class MetaResnet101(MetaModule):
    def __init__(self, num_class=64, inc=2048, temp=0.05):
        super(MetaResnet101, self).__init__()
        self.G = resnet101()
        del self.G.fc
        self.F1 = Predictor_deep(num_class=num_class, inc=inc, temp=temp)
        weights_init(self.F1)

    def forward(self, x):
        x = self.G(x)
        x = self.F1(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, inc=4096, ndomains=1, inc_mid=512, nclasses=1, projection=False):
        super(Discriminator, self).__init__()

        if projection:
            fin_size = inc_mid
        else:
            fin_size = ndomains
        self.fc1_1 = nn.Linear(inc, 1024)
        self.bn1_1 = nn.BatchNorm1d(1024)
        self.relu1_1 = nn.ReLU(True)
        self.fc2_1 = nn.Linear(1024, fin_size)

        self.log_soft = nn.LogSoftmax(dim=1)
        self.projection = projection

        if self.projection:
            self.V = map_module(nclasses, inc_mid)
            self.Psi = map_module(inc_mid, 1)

    def forward(self, x, reverse=True, alpha=1.0, label=None):

        if reverse:
            x = grad_reverse(x, alpha)
        x = self.relu1_1(self.bn1_1(self.fc1_1(x)))
        x = self.fc2_1(x)

        if self.projection:
            scalar_p = self.Psi(x)
            V_y = self.V(label)
            scalar_v = torch.bmm(V_y.unsqueeze(1), x.unsqueeze(2)).squeeze(2)
            out = scalar_p + scalar_v
        else:
            out = x
        out = self.log_soft(out)
        return out