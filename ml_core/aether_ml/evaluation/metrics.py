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
