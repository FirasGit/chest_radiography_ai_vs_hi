import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.fc = nn.Linear(512, num_classes)
        self.resnet18.fc = self.fc

    def forward(self, x):
        out = self.resnet18(x)
        return out
