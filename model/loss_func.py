import nnabla.functions as F


def l1_loss(pred_depth, gt_depth):
    valid_mask = F.greater_scalar(gt_depth, val=0.0)
    loss = F.sum(F.abs(pred_depth - gt_depth) * valid_mask) / F.sum(valid_mask)
    return loss
