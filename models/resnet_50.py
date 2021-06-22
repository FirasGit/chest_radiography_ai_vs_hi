import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.fc = nn.Linear(2048, num_classes)
        self.resnet50.fc = self.fc

    def forward(self, x):
        out = self.resnet50(x)
        return out
