import numpy as np
import random


def horizontal_flip(data, label, label_names, prob):
    _data = data.copy()
    _label = label.copy()
    _label_names = label_names.copy()
    if random.uniform(0, 1) < prob:
        _label = flip_labels(_label, _label_names)
        _data = np.flip(_data, axis=1).copy()
    return _data, _label


def horizontal_flip_situs_inversus(data, prob):
    _data = data.copy()
    _label_names = ['situs_inversus']
    if random.uniform(0, 1) < prob:
        _data = np.flip(_data, axis=1).copy()
        _label = [1]
    else:
        _label = [0]
    return _data, _label, _label_names


def flip_labels(_label, _label_names):
    labels_to_flip = ['Pleuraerguss', 'Infiltrate',
                      'Belstörungen', 'Belstörungenidem', 'Pneumothorax']
    for label_to_flip in labels_to_flip:
        all_left_label_idx = [_label_names.index(
            label_name) for label_name in _label_names if label_name.startswith(label_to_flip+'_li')]
        all_right_label_idx = [_label_names.index(
            label_name) for label_name in _label_names if label_name.startswith(label_to_flip+'_re')]
        for label_idx in range(len(all_left_label_idx)):
            left_label_idx = all_left_label_idx[label_idx]
            right_label_idx = all_right_label_idx[label_idx]
            tmp = _label[left_label_idx]
            _label[left_label_idx] = _label[right_label_idx]
            _label[right_label_idx] = tmp
    return _label
