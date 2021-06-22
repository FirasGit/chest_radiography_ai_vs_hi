import pytorch_lightning as pl
import torch.nn as nn
import torch
from typing import List, Any
import numpy as np
from metrics.roc_auc_score import roc_auc_score
from evaluation import create_cam_images, create_labels_csv


class ChestRadiographyClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_fnc, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fnc = loss_fnc
        self.optimizer = self.get_optimizer()
        self.scheduler, self.scheduler_metric = self.get_scheduler()
        self.metrics = {'AUC': roc_auc_score}   # TODO: Inject this

    def forward(self, x: torch.Tensor, **kwargs):
        return self.model(x, **kwargs)

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler, 'monitor': self.scheduler_metric}

    def _step(self, batch, prefix: str) -> dict:
        data, label, weights, label_names = batch['data'], batch['label'], batch['weights'], batch["label_names"]
        self.label_names = [elem[0] for elem in label_names]
        pred = self(data)
        loss = self.loss_fnc(pred, label)
        self.log(f'{prefix}/Step/Loss', loss)
        pred_prob = self._get_prediction_prob(pred)
        if prefix == 'Test':
            acquisition_numbers = batch['acquisition_numbers']
            create_cam_images(self, data, label, weights,
                              self.label_names, acquisition_numbers)
            create_labels_csv(self, label, pred_prob,
                              self.label_names, acquisition_numbers, self.cfg)
        return {'loss': loss, 'prediction': pred_prob, 'label': label}

    def _get_prediction_prob(self, pred):
        return torch.sigmoid(pred)

    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch, 'Train')

    def validation_step(self, val_batch, batch_idx):
        return self._step(val_batch, 'Val')

    def test_step(self, test_batch, batch_idx):
        return self._step(test_batch, 'Test')

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self._epoch_end(outputs, 'Train')

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self._epoch_end(outputs, 'Val')

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self._epoch_end(outputs, 'Test')

    def _epoch_end(self, outputs: List[Any], prefix: str):
        epoch_loss = torch.stack([value['loss'] for value in outputs]).mean()
        predictions_for_all_classes, labels_for_all_classes = \
            self.get_sequential_prediction_and_labels_all_classes(outputs)

        metric_values = {}
        num_classes = len(predictions_for_all_classes)
        for class_idx in range(num_classes):
            metric_values.update({metric + '_' + self.label_names[class_idx]:
                                  fn(labels_for_all_classes[class_idx],
                                     predictions_for_all_classes[class_idx])
                                  for metric, fn in self.metrics.items()})

        self.logger.experiment.add_scalar(
            f'{prefix}/Epoch/Loss', epoch_loss, self.current_epoch)

        for key, value in metric_values.items():
            self.logger.experiment.add_scalar(
                f'{prefix}/Epoch/{key}', value, self.current_epoch)

        mean_auc_over_all_classes = np.array(
            list(metric_values.values())).mean()
        self.logger.experiment.add_scalar(
            f'{prefix}/Epoch/Mean AUC', mean_auc_over_all_classes, self.current_epoch)

        # Log for checkpoint monitoring
        self.log(f'{prefix}_epoch_loss', epoch_loss, self.current_epoch)
        self.log(f'{prefix}_mean_auc',
                 mean_auc_over_all_classes, self.current_epoch)

    def get_sequential_prediction_and_labels_all_classes(self, outputs):
        aggregated = {}
        num_classes = len(outputs[0]['prediction'][0])

        # create list of lists where each sublist contains the class-wise predictions for each sample
        # e.g. [ s1-[cA, cB, cC], s2-[cA, cB, cC], s3-[cA, cB, cC] ] such that here the first sublist
        # contains all predictions from the first sample for all classes (s1 stands for sample 1)
        # len(vals['key']) defines the batch_size of the current step
        for key in ['prediction', 'label']:
            val = [vals[key][sample_idx, :].tolist()
                   for vals in outputs for sample_idx in range(len(vals[key]))]
            aggregated[key] = val

        # create list of lists where each sublist contains the class-wise predictions for all samples
        # e.g. [ [cA_s1, cA_s2, cA_s3], [cB_s1, cB_s2, cB_s3] ]
        predictions_for_all_classes = [torch.Tensor(aggregated['prediction'])[:, class_idx].tolist()
                                       for class_idx in range(num_classes)]
        labels_for_all_classes = [torch.Tensor(aggregated['label'])[:, class_idx].tolist()
                                  for class_idx in range(num_classes)]

        return predictions_for_all_classes, labels_for_all_classes

    def get_optimizer(self):
        if self.cfg.optimizer.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), self.cfg.optimizer.learning_rate)

        return optimizer

    def get_scheduler(self):
        if self.cfg.scheduler.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=self.cfg.scheduler.patience, factor=self.cfg.scheduler.scheduler_factor,
                threshold=1e-4, verbose=True)
            scheduler_metric = self.cfg.scheduler.scheduler_metric

        elif self.cfg.scheduler.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.cfg.scheduler.step_size, gamma=self.cfg.scheduler.gamma)
            scheduler_metric = None

        return scheduler, scheduler_metric
