from datasets.chest_radiography_uka import ChestRadiographyUKA
from torch.utils.data import DataLoader
from training.transforms import get_transforms


def get_dataloader(cfg):
    train_transforms, val_transforms = get_transforms(cfg)
    train_dataset = ChestRadiographyUKA('train', cfg.annotations.path_to_train_annotation_csv,
                                        cfg.data.path_to_data_dir,
                                        cfg,
                                        transforms=train_transforms)
    train_dataloader = DataLoader(train_dataset,
                                  cfg.meta.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=False)

    val_dataset = ChestRadiographyUKA('valid', cfg.annotations.path_to_valid_annotation_csv, cfg.data.path_to_data_dir, cfg,
                                      transforms=val_transforms)
    val_dataloader = DataLoader(
        val_dataset, cfg.meta.batch_size, shuffle=False, num_workers=8, drop_last=False)

    test_dataset = ChestRadiographyUKA('test', cfg.annotations.path_to_test_annotation_csv, cfg.data.path_to_data_dir, cfg,
                                       transforms=val_transforms)
    test_dataloader = DataLoader(
        test_dataset, cfg.meta.batch_size, shuffle=False, num_workers=8, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader
