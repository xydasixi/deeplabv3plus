import torch
import torch.nn as nn
from model.xception import Modified_Aligned_Xception
from model.ASPP import ASPP
from model.decoder import Decoder

class DeeplabV3_plus(nn.Module):
    """
    根据backbone--改进的xception，ASPP模块，decoder模块组成Deeplabv3+模型
    num_classes表示包含背景的类别数量
    """
    def __init__(self, num_classes):
        super(DeeplabV3_plus, self).__init__()
        self.backbone = Modified_Aligned_Xception()
        self.aspp = ASPP(in_channels=2048, out_channels = 256)
        self.decoder = Decoder(in_channels1 = 128, out_channels1 = 48, in_channels2 = 304, out_channels2 = 256 , num_classes = num_classes)
    def forward(self, input):
        low_level_features, x = self.backbone(input)
        x = self.aspp(x)
        out = self.decoder(low_level_features, x)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# if __name__ == "__main__":
#     model = DeeplabV3_plus()
#     input = torch.rand(1, 3, 512, 512)
#     output = model(input)
#     print(output.size())
