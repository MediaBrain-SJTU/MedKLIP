import torch
import numpy as np
import math
def f2_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 2, threshold)


def dice(logit, truth, threshold=0.5):
    batch_size = len(truth)

    with torch.no_grad():
        logit = logit.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert (logit.shape == truth.shape)

        probability = logit
        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)
        # print(len(neg_index), len(pos_index))

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


def mIoU(results, masks, threshold=0.5, eps=1):
    """
    :param results: b w h
    :param masks: b w h
    :param threshold: 0.5
    :return:
    """
    results = (results > threshold).float()
    b = len(results)
    results = results.view(b, -1).detach().cpu().numpy()
    masks = masks.view(b, -1).detach().cpu().numpy()
    intersection = (masks * results).sum(1)
    union = results.sum(1) + masks.sum(1) - intersection
    union = torch.from_numpy(union)
    intersection = torch.from_numpy(intersection)
    intersection = torch.clamp(intersection, 0, 1e8)
    union = torch.clamp(union, 0, 1e8)
    ious = ((intersection + eps) / (union + eps)).mean()
    return ious


def fbeta_score_threshold_matrix(y_true, y_pred, threshold, beta=2, eps=1e-9):
    beta2 = beta ** 2
    threshold = torch.from_numpy(threshold)
    threshold = threshold.unsqueeze(0).repeat(len(y_pred), 1).float()
    y_pred = (y_pred > threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision * recall).
            div(precision.mul(beta2) + recall + eps).
            mul(1 + beta2)).item()


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta ** 2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision * recall).
            div(precision.mul(beta2) + recall + eps).
            mul(1 + beta2)).item()


def create_class_weight(labels_dict, mu=0.5):
    total = np.sum(np.array(list(labels_dict.values())))
    keys = labels_dict.keys()
    class_weight = dict()
    class_weight_log = dict()
    for key in keys:
        score = total / float(labels_dict[key])
        score_log = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = round(score, 2) if score > 1.0 else round(1.0, 2)
        class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else round(1.0, 2)

    return class_weight, class_weight_log


def get_weight(targets):
    dict_label = {}
    for Target in targets:
        labels = Target.split(' ')
        labels = [int(l) for l in labels]
        for label in labels:
            if label in dict_label.keys():
                dict_label[label] += 1
            else:
                dict_label[label] = 1
    # n_labels = sum(list(dict_label.values()))
    Target2 = [sorted([int(i) for i in t.split(' ')]) for t in targets]
    class_weight, class_weight_log = create_class_weight(dict_label)
    # prob_dict, prob_dict_bal = cls_wts(dict_label, n_labels)
    weights = []
    for t in Target2:
        w = 0
        for t_ in t:
            w += class_weight_log[t_]
        weights.append(w)
    return weights
