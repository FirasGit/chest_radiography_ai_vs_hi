from torch.nn import BCEWithLogitsLoss


class BCEWithLogitsLossIgnoreIndex():
    def __init__(self, ignore_index, *args, **kwargs):
        self.ignore_index = ignore_index
        self.loss_fnc = BCEWithLogitsLoss(*args, **kwargs)

    def __call__(self, pred, target):
        mask = target == self.ignore_index
        return self.loss_fnc(pred[~mask], target[~mask])
