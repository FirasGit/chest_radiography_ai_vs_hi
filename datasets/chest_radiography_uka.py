from torch.utils.data import Dataset
import pandas as pd
import SimpleITK as sitk
import os
import numpy as np
from typing import Dict, Sequence
from omegaconf import DictConfig
import torch
from augmentations.horizontal_flip import horizontal_flip, horizontal_flip_situs_inversus


def read_csv_file(path_to_csv: str) -> pd.DataFrame:
    return pd.read_csv(path_to_csv)


def reformat_data(data: np.ndarray) -> np.ndarray:
    """
    Function to reformat the numpy array after reading out the nifti. The problem is two-fold:
        1. The nifti datatype is numpy.uint16
        2. The classification network typically expects an image with three channels (e.g. RGB)
    :param data: The data to transform
    :return: The transformed data
    """
    # Type Float32 is important because model weights are of type torch.cuda.FloatTensor not double
    data = np.repeat(data[:, :, np.newaxis], 3, axis=2).astype('float32')
    return data


class ChestRadiographyUKA(Dataset):
    """
    Dataset class that loads the annotations for the Chest Radiography images of the UKAachen
    """

    def __init__(self, split: str, path_to_annotation_csv: str, path_to_data_dir: str, cfg: DictConfig,
                 transforms: None, weights: Sequence = None):
        self.split = split
        self.path_to_annotation_csv = path_to_annotation_csv
        self.path_to_data_dir = path_to_data_dir
        self.annotations = read_csv_file(self.path_to_annotation_csv)
        self.cfg = cfg
        self.transforms = transforms
        self.weights = weights

    def _get_data(self, idx: int) -> np.ndarray:
        accession_number = str(self.annotations.iloc[idx][1])
        path_to_nifti_file = os.path.join(
            self.path_to_data_dir, str(accession_number) + '.nii')
        sitk_img = sitk.ReadImage(path_to_nifti_file)
        _data = sitk.GetArrayFromImage(sitk_img)
        data = reformat_data(_data)
        return data

    def _transform_data(self, data):
        return self.transforms(data) if self.transforms else torch.FloatTensor(data)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        label = self.annotations.iloc[idx][11:].to_numpy().astype(
            float)  # Ignore the meta data in the first 5 columns
        label_names = self.annotations.columns[11:].to_list()
        _data = self._get_data(idx)

        if self.cfg.situs_inversus.train is True:
            _data, label, label_names = horizontal_flip_situs_inversus(
                _data, prob=self.cfg.situs_inversus.prob)
            data = self._transform_data(_data)
            # self.weights # TODO: Pytorch Lightning can't work with None values enable self.weights later
            weights = [0]
        else:
            if self.split == 'train':
                _data, label = horizontal_flip(
                    _data, label, label_names, prob=self.cfg.augmentation.horizontal_flipping.prob)
            data = self._transform_data(_data)
            # self.weights # TODO: Pytorch Lightning can't work with None values enable self.weights later
            weights = [0]

        return {'data': data,
                'label': label,
                'weights': weights,
                'label_names': label_names,
                'acquisition_numbers': self.annotations.iloc[idx]['Anforderungsnummer']}
