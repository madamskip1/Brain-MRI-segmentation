from torch import nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets, smooth=1.0):
        return 1.0 - DiceLoss.__dice_coef(outputs, targets, smooth)

    @staticmethod
    def __dice_coef(outputs, targets, smooth):
        __outputs = outputs.view(-1)
        __targets = targets.view(-1)
        intersection = (__outputs * __targets).sum()
        dice = (2.0 * intersection + smooth) / (__outputs.sum() + __targets.sum() + smooth)

        return dice
