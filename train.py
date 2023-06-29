import torch
import model
import loader
import loss
from torch.utils import data
import numpy as np
from torch import nn
from argparse import ArgumentParser
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
batch_size = 5
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

net = model.Net.LightNet(pretrained=False).to(device)
optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch+1, eta_min=1e-7)
print("Normal train mode with pretrained model")


loss = loss.MyLoss()

"""Print model's information"""
# summary(net, input_size=(3, 224, 224), device='cpu')
# exit()


def train(net, device, train_loader, loss, optimizer, epoch_num, scheduler):
    """Train function"""
    net.train()
    sod_loss = []

    for batch_idx, dict in enumerate(train_loader):
        x = dict['image'].to(device)
        label = dict['label'].to(device)
        optimizer.zero_grad()

        # Training
        y, y2, y3, y4 = net(x)  # output shape [batch_size,1,224,224]

        loss_t = loss(y, y2, y3, y4, label)

        loss_t.backward()

        optimizer.step()

        sod_loss.append(loss_t.item())

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  batch_loss:{:.6f}'.format(
                epoch_num, (batch_idx+1) * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_t.item(),
            ))

    sod_loss = sum(sod_loss) / len(sod_loss)

    print("Epoch: {} loss:{:.6f} lr:{:.8f}".format(
        epoch_num,
        sod_loss,
        optimizer.param_groups[0]['lr'],
    ))

    torch.save(net.state_dict(), "./checkpoints/x-ray.pth")
    scheduler.step()  # 执行学习率调整策略


"""Start training"""
for epoch_num in range(1, epoch + 1):
    train(net, device, train_data_loader, loss, optimizer, epoch_num, scheduler)
