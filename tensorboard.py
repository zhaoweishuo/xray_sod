from tensorboardX import SummaryWriter
import loader
from torch.utils import data
from model import *
import cv2
import numpy as np
import torch

writer = SummaryWriter('./tb')  # 建立一个保存数据用的目录，save是输出的文件名


def hook_func(net, input):
    """
    使用torch.nn.Module.register_forward_hook(hook_func)函数可以实现特征图的可视化，
    register_forward_hook是一个钩子函数，设置完后，当输入图片进行前向传播的时候就会执行自定的函数，该函数作为参数传到register_forward_hook方法中。
    hook_func函数可从前向过程中接收到三个参数：hook_func(module, input, output)。其中module指的是模块的名称，比如对于ReLU模块，module是ReLU()，
    对于卷积模块，module是Conv2d(in_channel=…)，注意module带有具体的参数。input和output就是我们需要的特征图，
    这二者分别是module的输入和输出，输入可能有多个（比如concate层就有多个输入），输出只有一个，所以input是一个tuple，其中每一个元素都是一个Tensor，而输出就是一个Tensor。
    一般而言output可能更经常拿来做分析。我们可以在hook_func中将特征图画出来并保存为图片，所以hook_func就是我们实现可视化的关键。

    tensorboard --logdir tb  来启动tensorboard查看特征图
    ps:tensorboard生成图片太小，还是手动保存
    """

    map = input[0][0]
    map = map.unsqueeze(1)
    map = map.numpy()
    global i
    writer.add_images("feature1", map, i, dataformats='NCHW')
    i += 1

    # global i
    # map = input[0][0]
    # map = torch.sigmoid(map)*255
    # map = map.numpy().astype(np.uint8)
    # index = 0
    # for pic in map:
    #     resize = cv2.resize(pic, (224, 224))
    #     resize.astype('uint8')
    #     cv2.imwrite("./processImg/"+str(i)+"/"+str(index)+".png", resize)
    #     index += 1
    # i += 1




data_root = "./dataset/test/SOD/"
batch_size = 1
device = torch.device("cpu")

test_dataset = loader.LoadTestDataset(data_root=data_root)
test_data_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )


net = Net.LightNet(pretrained=False).to(device)

useful_layer = {'encoder.layer1.0.conv.2',
                'encoder.layer2.0.conv.1.conv.2',
                'encoder.layer3.0.conv.1.conv.2',
                'encoder.layer4.0.conv.1.conv.2',
                'encoder.layer5.0.conv.1.conv.2'}

for name, m in net.named_modules():

    if name in useful_layer:
        # 符合条件的层数触发钩子函数
        m.register_forward_pre_hook(hook_func)

i = 0
net.eval()
with torch.no_grad():
    for batch_idx, dict in enumerate(test_data_loader):
        x = dict['image'].to(device)
        print(dict['name'])
        out, _, _, _ = net(x)
        # 运行一张图就停止
        exit()
