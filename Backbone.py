import torch.nn as nn
import torchvision

class ResNetFeatures(nn.Module):
    def __init__(self):
        super(ResNetFeatures, self).__init__()

        model = torchvision.models.resnet50(pretrained=True)
        self.seq1 = nn.Sequential(model.conv1,
                                  model.bn1,
                                  model.relu,
                                  model.maxpool,
                                  model.layer1,
                                  model.layer2,
                                  model.layer3)

        self.out_channels = 1024

    def forward(self, x):
        x = self.seq1(x)

        return x