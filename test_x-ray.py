import torch
import loader
from torch.utils import data
import cv2
from model import *
import time


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


batch_size = 8
# device = torch.device('cpu')
device = try_gpu(1)
data_root = "./dataset/test/x-ray-image/"

test_dataset = loader.LoadTestDataset(data_root=data_root)
test_data_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )


net = Net.LightNet(pretrained=False).to(device)
net.load_state_dict(torch.load("./checkpoints/x-ray-self.pth", map_location=device))  # 加载训练好的模型


def test(model, test_loader, device=None):
    model.eval()  # 测试过程中会使用model.eval()，这时神经网络会沿用batch normalization的值，并不使用dropout

    # 测试时不计算梯度
    with torch.no_grad():
        start = time.process_time()
        for batch_idx, dict in enumerate(test_loader):
            x = dict['image'].to(device)
            name = dict['name']
            w = dict['w']
            h = dict['h']

            out, _, _, _ = model(x)
            image = out.squeeze(0).cpu().numpy()  # 去掉索引为0的空维度
            image *= 255



            for index, _ in enumerate(name):
                # 图片数据维度为4 代表bitchsize大于1
                if image.ndim == 4:
                    resize = cv2.resize(image[index][0], (int(w[index]), int(h[index])))
                else:
                    resize = cv2.resize(image[0], (int(w[index]), int(h[index])))

                resize.astype('uint8')

                cv2.imwrite(data_root+"predict/"+name[index]+".png", resize)


            print("saved {}/{}".format((batch_idx+1)*len(x), len(test_loader.dataset)))
        end = time.process_time()
        print("done！ takes {} seconds".format(end-start))


test(net, test_data_loader, device)
