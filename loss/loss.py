import torch.nn as nn
import torch
import torch.nn.functional as F

"""My Loss Function"""


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.iou = IOU()

    def forward(self, y, y2, y3, y4, label):
        loss_y = self.bce(y, label) + self.iou(y, label)
        loss_y2 = self.bce(y2, label) + self.iou(y2, label)
        loss_y3 = self.bce(y3, label) + self.iou(y3, label)
        loss_y4 = self.bce(y4, label) + self.iou(y4, label)
        return loss_y+loss_y2+loss_y3+loss_y4


class FinalLoss(nn.Module):
    def __init__(self):
        super(FinalLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.iou = IOU()

    def forward(self, y, label):
        loss = self.bce(y, label) + self.iou(y, label)
        return loss


class IOU(torch.nn.Module):
    """IOU loss function"""
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1

            # IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)

        return IoU / b
