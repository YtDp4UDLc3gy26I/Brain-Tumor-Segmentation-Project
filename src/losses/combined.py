import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_dice_no_bg(logits, targets, eps: float = 1e-6):
    """
    Soft Dice over classes 1..C-1 (ignore background=0).
    logits: [B,C,H,W], targets: [B,H,W] ints in [0..C-1]
    returns Dice loss = (1 - mean Dice)
    """
    C = logits.shape[1]
    prob   = torch.softmax(logits, dim=1)                     # [B,C,H,W]
    onehot = F.one_hot(targets, C).permute(0, 3, 1, 2).float()# [B,C,H,W]
    prob_nb   = prob[:, 1:]     # drop background
    onehot_nb = onehot[:, 1:]
    inter  = (prob_nb * onehot_nb).sum(dim=(0, 2, 3))
    denom  = prob_nb.sum(dim=(0, 2, 3)) + onehot_nb.sum(dim=(0, 2, 3))
    dice_c = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice_c.mean()

class CombinedLoss(nn.Module):
    """ total = alpha * CrossEntropy + beta * SoftDice(no-bg) """
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce   = self.ce(logits, targets)
        dice = soft_dice_no_bg(logits, targets)
        return self.alpha * ce + self.beta * dice