import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Decoder(nn.Module):
    def __init__(self, in_channels1, out_channels1, in_channels2, out_channels2, num_classes):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=in_channels1, out_channels=out_channels1, kernel_size=1, padding=0,bias=False),
                                   nn.BatchNorm2d(num_features=out_channels1),
                                   nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=in_channels2, out_channels=out_channels2, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=out_channels2),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(in_channels=out_channels2, out_channels=out_channels2, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=out_channels2),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.1),
                                   nn.Conv2d(in_channels=out_channels2, out_channels=num_classes, kernel_size=1, padding=0, bias=False)
                                   )
        self._initialize_weights()
    def forward(self, x1, x2 ):
        y1 = self.layer1(x1)
        y2 = F.interpolate(x2, size=x1.size()[2:], mode = 'binear', align_corners=True)
        out = self.layer2(torch.cat((y1,y2),1))
        out = F.interpolate(out, scale_factor=4, mode = 'binear', align_corners=True)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()