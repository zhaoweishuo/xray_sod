import loader
from torch.utils import data
import cv2
import model
import torch
import time
import numpy as np


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


batch_size = 1
# device = torch.device('cpu')
device = try_gpu()
data_root = "./dataset/test/x-ray-image/"

test_dataset = loader.LoadTestDataset(data_root=data_root)
test_data_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )


net = model.MobileNet(pretrained=False).to(device)
net.load_state_dict(torch.load("./checkpoints/x-ray-mobilenet.pth", map_location=device))  # 加载训练好的模型


def test(model, test_loader, device=None):
    model.eval()  # 测试过程中会使用model.eval()，这时神经网络会沿用batch normalization的值，并不使用dropout
    start = time.process_time()
    # 测试时不计算梯度
    with torch.no_grad():
        for batch_idx, dict in enumerate(test_loader):
            x = dict['image'].to(device)
            name = dict['name']
            w = dict['w']
            h = dict['h']

            out = model(x)
            image = out.squeeze(0).cpu().numpy()  # 去掉索引为0的空维度
            image *= 255

            image[image>50] += 100


            for index, _ in enumerate(name):
                resize = cv2.resize(image[index], (int(w[index]), int(h[index])))
                resize.astype('uint8')
                cv2.imwrite(data_root+"predict/"+name[index]+".png", resize)

            print("saved {}/{}".format((batch_idx+1)*len(x), len(test_loader.dataset)))
        end = time.process_time()
        print("done！ takes {} seconds".format(end-start))


test(net, test_data_loader, device)
