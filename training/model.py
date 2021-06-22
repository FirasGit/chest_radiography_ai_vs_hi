from models import ResNet18, ResNet34, ResNet50


def get_model(cfg):
    if cfg.model.name == 'resnet_18':
        return ResNet18(cfg.meta.num_classes)
    elif cfg.model.name == 'resnet_34':
        return ResNet34(cfg.meta.num_classes)
    elif cfg.model.name == 'resnet_50':
        return ResNet50(cfg.meta.num_classes)
    else:
        raise KeyError
