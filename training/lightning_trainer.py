import pytorch_lightning as pl
from training.lightning_classifier import ChestRadiographyClassifier
import hydra
import os
from training.dataloader import get_dataloader
from omegaconf import DictConfig
from training.best_checkpoint import get_best_checkpoint
from training.loss_function import get_loss_fnc
from training.model import get_model


@hydra.main(config_path='../configs/train_config', config_name="base_cfg")
def run(cfg: DictConfig):
    # Set path where to save the checkpoints and then retrieve the best checkpoint if available
    checkpoint_dir = os.path.join(cfg.meta.output_dir, cfg.meta.prefix_name)
    checkpoint_callback = get_checkpoint_callback(checkpoint_dir, cfg)
    resume_from_checkpoint = get_best_checkpoint(
        checkpoint_dir, metric=cfg.checkpoint.monitor, mode=cfg.checkpoint.mode)

    # Allow monitoring of the learning rate
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # Make PyTorch Lightning stop early when given metric doesn't change a lot
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor=cfg.early_stopping.monitor,
                                                         patience=cfg.early_stopping.patience,
                                                         mode=cfg.early_stopping.mode,
                                                         min_delta=cfg.early_stopping.min_delta)
    model = get_model(cfg)
    loss_fnc = get_loss_fnc(cfg)

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(cfg)
    lightning_model = ChestRadiographyClassifier(model=model,
                                                 loss_fnc=loss_fnc,
                                                 cfg=cfg)
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.meta.epochs, callbacks=[lr_logger, early_stopping_callback],
                         checkpoint_callback=checkpoint_callback, resume_from_checkpoint=resume_from_checkpoint)
    if cfg.meta.test is True:
        trainer.test(lightning_model, test_dataloaders=test_dataloader)
    else:
        trainer.fit(lightning_model, train_dataloader=train_dataloader,
                    val_dataloaders=val_dataloader)


def get_checkpoint_callback(checkpoint_dir, cfg):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir,
                                                       filename=cfg.checkpoint.filename,
                                                       monitor=cfg.checkpoint.monitor,
                                                       mode=cfg.checkpoint.mode)
    return checkpoint_callback


if __name__ == '__main__':
    run()
