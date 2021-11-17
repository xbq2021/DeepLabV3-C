import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, layers, output_stride, BatchNorm, pretrained=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        resnet = models.resnet34(pretrained=True)

        resnet = models.resnet34(pretrained=True)
        self.firstconv1 = resnet.conv1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.conv3 = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, input):

        xx = self.firstconv1(input)
        x2 = self.conv1(input)
        x3 = self.conv2(x2)
        x = self.conv2(x3)

        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.conv3(x)
        e1 = self.encoder1(x)
        low_level_feat =e1
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        return e4, low_level_feat

def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model
