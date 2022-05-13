def calculate_accuracy(outputs, targets, mask_true_threshold, smooth=1.0):
    __outputs = (outputs > mask_true_threshold) * 1.0
    __outputs = __outputs.view(-1)
    __targets = targets.view(-1)

    intersection = (__outputs * __targets).sum()
    total = (__outputs + __targets).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return IoU
