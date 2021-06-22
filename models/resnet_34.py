import torch
import torch.nn as nn
import torchvision.models as models


class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        self.fc = nn.Linear(512, num_classes)
        self.resnet34.fc = self.fc

        # Gradient Placeholder for CAM
        self.gradient = None

    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activation(self, x):
        x = self.resnet34.conv1(x)
        x = self.resnet34.bn1(x)
        x = self.resnet34.relu(x)
        x = self.resnet34.maxpool(x)

        x = self.resnet34.layer1(x)
        x = self.resnet34.layer2(x)
        x = self.resnet34.layer3(x)
        x = self.resnet34.layer4(x)
        return x

    def forward(self, x, create_cam=False):
        x = self.get_activation(x)

        if create_cam is True:
            h = x.register_hook(self.activations_hook)

        x = self.resnet34.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet34.fc(x)
        return x
