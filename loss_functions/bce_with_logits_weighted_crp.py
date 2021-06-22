from torch.nn import BCEWithLogitsLoss
import torch


def is_between(value, min_val, max_val):
    return min_val <= value <= max_val


def get_infiltrate_weight(crp_value, infiltrate_target):
    # Do Lookup
    if (crp_value < 5) and (infiltrate_target == 0):
        return 2
    elif (is_between(crp_value, 5, 50)) and (infiltrate_target == 0):
        return 1
    elif (crp_value > 50) and (infiltrate_target == 0):
        return 0.5
    elif (crp_value < 5) and (infiltrate_target == 4):
        return 1
    elif (is_between(crp_value, 5, 50)) and (infiltrate_target == 4):
        return 2
    elif (crp_value > 50) and (infiltrate_target == 4):
        return 1
    elif (crp_value < 5) and (infiltrate_target in [1, 2, 3]):
        return 0.5
    elif (is_between(crp_value, 5, 50)) and (infiltrate_target in [1, 2, 3]):
        return 1
    elif (crp_value > 50) and (infiltrate_target in [1, 2, 3]):
        return 2


class BCEWithLogitsWeightedCRP():
    def __init__(self, ignore_index, *args, **kwargs):
        self.ignore_index = ignore_index
        self.loss_fnc = BCEWithLogitsLoss(reduction='none', *args, **kwargs)

    def __call__(self, pred, target):
        # TODO: Bad implementation. Assumes CRP to be at loc -2. Change this
        loss_target = target[:, :-2]
        unweighted_loss = self.loss_fnc(pred, loss_target)
        weight = []
        for batch_sample in target:
            crp_value = batch_sample[-2]
            sample_weight = torch.ones(len(batch_sample[:-2]))
            if crp_value != self.ignore_index:
                infiltrate_re_target = batch_sample[20:25].argmax()
                infiltrate_li_target = batch_sample[25:30].argmax()
                infiltrate_re_weight = get_infiltrate_weight(
                    crp_value, infiltrate_re_target)
                infiltrate_li_weight = get_infiltrate_weight(
                    crp_value, infiltrate_li_target)
                sample_weight[20:25] = infiltrate_re_weight
                sample_weight[25:30] = infiltrate_li_weight
            weight.append(sample_weight)

        weighted_loss = unweighted_loss * \
            torch.stack(weight).to(unweighted_loss.device)
        return weighted_loss.mean()
