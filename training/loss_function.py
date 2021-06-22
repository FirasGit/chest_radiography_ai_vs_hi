from torch.nn import BCEWithLogitsLoss
from loss_functions import BCEWithLogitsLossIgnoreIndex, BCEWithLogitsWeightedCRP


def get_loss_fnc(cfg):
    if cfg.optimizer.loss_fnc == "bcewithlogits":
        loss_fnc = BCEWithLogitsLoss()
    if cfg.optimizer.loss_fnc == "bcewithlogits_ignore_index":
        loss_fnc = BCEWithLogitsLossIgnoreIndex(ignore_index=99)
    if cfg.optimizer.loss_fnc == "bcewithlogits_weighted_crp":
        loss_fnc = BCEWithLogitsWeightedCRP(ignore_index=-1000)

    return loss_fnc
