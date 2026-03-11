import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        
        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice

class HybridLoss(nn.Module):
    """
    BCE + Dice Loss
    Helps segmentation models converge better on small regions where classes are imbalanced.
    """
    def __init__(self, bce_weight=0.6, dice_weight=0.4):
        super(HybridLoss, self).__init__()
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        dice = self.dice(preds, targets)
        
        return self.bce_weight * bce + self.dice_weight * dice

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss
    Designed to heavily penalize False Negatives (missing small objects like aircraft)
    more than False Positives (guessing an aircraft).
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (preds * targets).sum()
        FP = ((1 - targets) * preds).sum()
        FN = (targets * (1 - preds)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky

class HybridTverskyLoss(nn.Module):
    """
    BCE + Focal Tversky Loss
    Anchors the massive background to 0 using BCE while ruthlessly penalizing False Negatives on sparse foreground objects using Tversky.
    """
    def __init__(self, bce_weight=0.5, tversky_weight=0.5, alpha=0.7, beta=0.3, gamma=4/3):
        super(HybridTverskyLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        tversky = self.tversky(preds, targets)
        return self.bce_weight * bce + self.tversky_weight * tversky
