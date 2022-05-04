from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, smooth=1.0):
        return 1.0 - DiceLoss.__dice_coef(outputs, targets, smooth)

    @staticmethod
    def __dice_coef(outputs, targets, smooth):
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        intersection = (outputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)

        return dice
