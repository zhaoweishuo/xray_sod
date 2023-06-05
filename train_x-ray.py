import torch
import model
import loader
import loss
from torch.utils import data
import numpy as np
from torch import nn
import metric

from torchsummary import summary
"""
单独训练x光图像，不需要验证集来更新模型
"""

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
batch_size = 4
epoch = 60
device = try_gpu(0)
# same_seed(666)
lr = 3e-4
weight_decay = 1e-4

"""Loading data"""
train_dataset = loader.LoadTrainDataset(data_root="./dataset/x-ray-image/")
train_data_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

net = model.Net.LightNet(pretrained=False).to(device)
optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch+1, eta_min=1e-7)
loss = loss.MyLoss()

"""Print model's information"""
# summary(net, input_size=(3, 224, 224), device='cpu')
# exit()


def train(net, device, train_loader, loss, optimizer, epoch_num, scheduler):
    """Train function"""
    net.train()  # when you‘re training your model,enable Batch Normalization and Dropout
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

        if batch_idx % 4 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  batch_loss:{:.6f}'.format(
                epoch_num, (batch_idx+1) * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_t.item(),
            ))

    sod_loss = sum(sod_loss) / len(sod_loss)

    print("Epoch: {} loss:{:.6f} lr:{:.8f}".format(
        epoch_num,
        sod_loss,
        optimizer.param_groups[0]['lr']
    ))
    scheduler.step()  # 执行学习率调整策略

    torch.save(net.state_dict(), "./checkpoints/x-ray-self1.pth")  # 由于训练时间过长 每一轮存一次

    #  Save the log
    with open("./log/log_x-ray.txt", 'a') as file_object:
        file_object.write(
            "Epoch: {} loss:{:.6f} lr:{:.8f}\n".format(
                epoch_num,
                sod_loss,
                optimizer.param_groups[0]['lr'],
            ))


"""Start training"""
for epoch_num in range(1, epoch + 1):
    train(net, device, train_data_loader, loss, optimizer, epoch_num, scheduler)
