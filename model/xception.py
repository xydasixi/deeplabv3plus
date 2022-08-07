import numpy as np
import os, glob, datetime, time, math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, dilation = 1):
        super(Depthwise_Separable_Conv, self).__init__()
        self.Depthwise_Conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation = dilation,
            groups=in_channels,
            bias = False
        )
        self.Pointwise_Conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=1,
            bias=False
        )
    def forward(self, input):
        out = self.Depthwise_Conv(input)
        out = self.Pointwise_Conv(out)
        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num, stride=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_channels != in_channels:
            self.skip = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(num_features=out_channels))
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        layers = []

        channels = in_channels
        if not grow_first:
            layers += [nn.ReLU(inplace=True),
                       Depthwise_Separable_Conv(in_channels=in_channels, out_channels=in_channels),
                       nn.BatchNorm2d(num_features=in_channels)]

        layers += [nn.ReLU(inplace=True),
                   Depthwise_Separable_Conv(in_channels = in_channels, out_channels=out_channels),
                   nn.BatchNorm2d(num_features=out_channels)]
        channels = out_channels

        for i in range(num - 1):
            layers += [nn.ReLU(inplace=True),
                       Depthwise_Separable_Conv(in_channels=channels, out_channels=channels),
                       nn.BatchNorm2d(num_features=channels)]


        if not start_with_relu:
            layers = layers[1:]
        # else:
        #     layers[0] = nn.ReLU(inplace=False)

        if stride != 1:
            layers += [nn.ReLU(inplace=True),
                       Depthwise_Separable_Conv(in_channels=out_channels, out_channels=out_channels, stride=2),
                       nn.BatchNorm2d(num_features=out_channels)]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        out += skip
        return out
class Entry_Flow(nn.Module):
    def __init__(self):
        super(Entry_Flow, self).__init__()
        layer1 = [nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1,bias=False),
                  nn.BatchNorm2d(num_features=32),
                  nn.ReLU(inplace=True)]
        layer1 += [nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,bias=False),
                   nn.BatchNorm2d(num_features=64),
                   nn.ReLU(inplace=True)]
        self.layer1 = nn.Sequential(*layer1)

        self.layer2 = Block(in_channels=64, out_channels=128, num=2, stride=2, start_with_relu=False)
        self.layer3 = Block(in_channels=128, out_channels=256, num=2, stride=2)
        self.layer4 = Block(in_channels=256, out_channels=728, num=2, stride=2)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class Middle_Flow(nn.Module):
    def __init__(self):
        super(Middle_Flow, self).__init__()
        layers = []
        for i in range (16):
            layers += [Block(in_channels=728, out_channels=728, num=3)]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        return out

class Exit_Flow(nn.Module):
    def __init__(self):
        super(Exit_Flow, self).__init__()
        self.layer1 = Block(in_channels=728, out_channels=1024, num=2, stride=1, grow_first=False)
        layer2 = [Depthwise_Separable_Conv(in_channels=1024, out_channels=1536),
                  nn.BatchNorm2d(num_features=1536),
                  nn.ReLU(inplace=True)]
        layer2 += [Depthwise_Separable_Conv(in_channels=1536, out_channels=1536),
                  nn.BatchNorm2d(num_features=1536),
                  nn.ReLU(inplace=True)]
        layer2 += [Depthwise_Separable_Conv(in_channels=1536, out_channels=2048),
                  nn.BatchNorm2d(num_features=2048),
                  nn.ReLU(inplace=True)]
        self.layer2 = nn.Sequential(*layer2)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        return out

class Modified_Aligned_Xception(nn.Module):
    def __init__(self):
        super(Modified_Aligned_Xception, self).__init__()
        self.entry = Entry_Flow()
        self.middle = Middle_Flow()
        self.exit = Exit_Flow()
        self._initialize_weights()

    def forward(self, input):
        out = self.entry(input)
        out = self.middle(out)
        out = self.exit(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    model = Modified_Aligned_Xception()
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output.size())