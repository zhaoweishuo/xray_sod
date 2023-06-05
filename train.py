import torch
import model
import loader
import loss
from torch.utils import data
import numpy as np
from torch import nn
import metric
from argparse import ArgumentParser
import cv2
from torchsummary import summary

"""Input parameter"""
parser = ArgumentParser()
parser.add_argument('--mode', type=str, help='Train mode')
args = parser.parse_args()


"""Useful Function"""


def try_gpu(i=0):
    """Return gpu(i) or cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_weights(m):
    """Initialize weights, If you are not using a pretrained model"""
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)


"""Parameter setting"""
batch_size = 24
epoch = 60
device = try_gpu(1)
# same_seed(666)
lr = 3e-4
weight_decay = 1e-4

"""Loading data"""
train_dataset = loader.LoadTrainDataset(data_root="./dataset/train/")
train_data_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

val_dataset = loader.LoadValDataset(data_root="./dataset/val/")
val_data_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

"""Model & Optimizer"""
if args.mode == 'normal':
    net = model.Net.LightNet(pretrained="./pretrained/pretrain.pth").to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch+1, eta_min=1e-7)
    print("Normal train mode with pretrained model")
elif args.mode == 'refinement':
    net = model.Net.LightNet(pretrained=False).to(device)
    net.load_state_dict(torch.load("./checkpoints/x-ray.pth", map_location=device))  # 加载训练好的模型
    optimizer = torch.optim.Adam(params=net.parameters(), lr=3e-7, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    print("Refinement train mode with self-trained model")
else:
    exit("No parameter mode")

loss = loss.MyLoss()

"""Print model's information"""
summary(net, input_size=(3, 224, 224), device='cpu')
exit()


def train(net, device, train_loader, val_data_loader, loss, optimizer, epoch_num, scheduler):
    """Train function"""
    net.train()  # when you‘re training your model,enable Batch Normalization and Dropout
    global best_mae
    global best_maxf
    global best_meanf
    sod_loss = []

    for batch_idx, dict in enumerate(train_loader):
        x = dict['image'].to(device)
        label = dict['label'].to(device)
        optimizer.zero_grad()  # 梯度归零

        # Training
        y, y2, y3, y4 = net(x)  # output shape [batch_size,1,224,224]

        loss_t = loss(y, y2, y3, y4, label)

        loss_t.backward()

        optimizer.step()  # 更新权重要一起放最后

        sod_loss.append(loss_t.item())

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  batch_loss:{:.6f}'.format(
                epoch_num, (batch_idx+1) * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_t.item(),
            ))

    sod_loss = sum(sod_loss) / len(sod_loss)
    metric = val(net, device, epoch_num, val_data_loader)

    print("Epoch: {} loss:{:.6f} lr:{:.8f} mae:{:.4f} maxF:{:.4f} meanF:{:.4f}".format(
        epoch_num,
        sod_loss,
        optimizer.param_groups[0]['lr'],
        metric['MAE'],
        metric['MaxF'],
        metric['MeanF'],
    ))

    # 模型更新条件为验证集mae↓ maxF↑ meanF↑ 三个测评指标有其中至少两个有所改观
    sign1 = 1 if best_mae > metric['MAE'] else 0
    sign2 = 1 if best_maxf < metric['MaxF'] else 0
    sign3 = 1 if best_meanf < metric['MeanF'] else 0
    sign = sign1 + sign2 + sign3
    if sign >= 2:
        torch.save(net.state_dict(), "./checkpoints/x-ray.pth")  # 由于训练时间过长 每一轮存一次
        best_mae = metric['MAE']
        best_maxf = metric['MaxF']
        best_meanf = metric['MeanF']
        print("Better metric, Save the model")
    else:
        print("No better metric")

    scheduler.step()  # 执行学习率调整策略

    #  Save the log
    with open("./log/log.txt", 'a') as file_object:
        file_object.write(
            "Epoch: {} loss:{:.6f} lr:{:.8f} mae:{:.4f} maxF:{:.4f} meanF:{:.4f} sign:{}\n".format(
                epoch_num,
                sod_loss,
                optimizer.param_groups[0]['lr'],
                metric['MAE'],
                metric['MaxF'],
                metric['MeanF'],
                sign,
            ))


def val(model, device, epoch_num, val_loader):
    """Validation after train"""
    print("Start validation for epoch {}".format(epoch_num))
    model.eval()
    metric_obj = metric.CalTotalMetric(num=len(val_loader))

    with torch.no_grad():
        for batch_idx, dict in enumerate(val_loader):
            x = dict['image'].to(device)
            y_hat = dict['label'].to(device)

            out, _, _, _ = model(x)
            predict = out.squeeze().cpu().numpy()  # 去除多余维度
            gt = y_hat.squeeze().cpu().numpy()

            metric_obj.update(predict, gt)  # 计算评估指标

    return metric_obj.show()


"""Start training"""
# 微调模型时需要修改以下初始值
best_mae = 0.0493
best_maxf = 0.9158
best_meanf = 0.8582
for epoch_num in range(1, epoch + 1):
    train(net, device, train_data_loader, val_data_loader, loss, optimizer, epoch_num, scheduler)
