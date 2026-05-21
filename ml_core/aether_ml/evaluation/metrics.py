import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Standard focal Tversky formulation:
    TP / (TP + alpha * FP + beta * FN + eps)
    Higher beta penalizes missed foreground harder.
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0):
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
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky

class HybridTverskyLoss(nn.Module):
    """
    BCE + Focal Tversky Loss
    Anchors the massive background to 0 using BCE while ruthlessly penalizing False Negatives on sparse foreground objects using Tversky.
    """
    def __init__(self, bce_weight=0.4, tversky_weight=0.6, alpha=0.3, beta=0.7, gamma=0.75):
        super(HybridTverskyLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        tversky = self.tversky(preds, targets)
        return self.bce_weight * bce + self.tversky_weight * tversky


class SobelOperator(nn.Module):
    """
    Computes Sobel edge maps in 2D.
    """
    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.],
                                 [ 0.,  0.,  0.],
                                 [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.kernel_x = self.kernel_x.to(x.device, dtype=x.dtype)
        self.kernel_y = self.kernel_y.to(x.device, dtype=x.dtype)
        
        x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(x_padded, self.kernel_x)
        grad_y = F.conv2d(x_padded, self.kernel_y)
        
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return torch.clamp(magnitude, 0.0, 1.0)


class HybridTverskyBoundaryLoss(nn.Module):
    """
    Hybrid Tversky Loss with an added boundary-aware loss component:
    L_boundary = BCE(sobel(pred_prob), sobel(label))
    Combined loss: 0.7 * L_tversky + 0.3 * L_boundary
    """
    def __init__(self, bce_weight: float = 0.4, tversky_weight: float = 0.6, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75) -> None:
        super().__init__()
        self.hybrid_tversky = HybridTverskyLoss(
            bce_weight=bce_weight,
            tversky_weight=tversky_weight,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        self.sobel = SobelOperator()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        L_tversky = self.hybrid_tversky(preds, targets)
        
        pred_prob = torch.sigmoid(preds)
        sobel_pred = self.sobel(pred_prob)
        sobel_label = self.sobel(targets)
        
        # Clamp inputs for binary cross entropy to avoid NaNs
        sobel_pred = torch.clamp(sobel_pred, 1e-7, 1.0 - 1e-7)
        
        # Disable autocast for BCE to avoid PyTorch's safety checks / RuntimeErrors under mixed precision
        with torch.amp.autocast('cuda', enabled=False):
            L_boundary = F.binary_cross_entropy(sobel_pred.float(), sobel_label.float())
        
        return 0.7 * L_tversky + 0.3 * L_boundary.to(L_tversky.dtype)

