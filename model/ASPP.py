import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ASPP(nn.Module):
    """
    ASPP模块
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.pyramid1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size = 1, bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True)
                                     )
        self.pyramid2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True)
                                     )
        self.pyramid3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=12, dilation=12,bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True)
                                     )
        self.pyramid4 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=18, dilation=18,bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True)
                                     )
        self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(num_features=out_channels),
                                     nn.ReLU(inplace=True)
                                    )
        self.output = nn.Sequential(nn.Conv2d(in_channels=5*out_channels, out_channels=out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.5)
                                   )
        self._initialize_weights()

    def forward(self, input):
        y1 = self.pyramid1(input)
        y2 = self.pyramid2(input)
        y3 = self.pyramid3(input)
        y4 = self.pyramid4(input)
        y5 = F.interpolate(self.pooling(input), size=y4.size()[2:], mode='bilinear', align_corners=True)
        out = self.output(torch.cat([y1,y2,y3,y4,y5],1))
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# if __name__ == "__main__":
#     model = ASPP(in_channels=2048, out_channels = 256)
#     input = torch.rand(1, 2048, 32, 32)
#     output = model(input)
#     print(output.size())