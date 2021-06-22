from torchvision import transforms
from omegaconf import DictConfig


def get_transforms(cfg: DictConfig):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=cfg.augmentation.rotation.degrees),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transforms, val_transforms
