import torch
import loader
from torch.utils import data
import cv2
from model import *
import time


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


batch_size = 1
# device = torch.device('cpu')
device = try_gpu()
data_root = "./dataset/test/SOD/"

test_dataset = loader.LoadTestDataset(data_root=data_root)
test_data_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )


net = Net.LightNet(pretrained=False).to(device)
net.load_state_dict(torch.load("./checkpoints/x-ray.pth", map_location=device))  # 加载训练好的模型


def test(model, test_loader, device=None):
    model.eval()
    start = time.process_time()

    with torch.no_grad():
        for batch_idx, dict in enumerate(test_loader):
            x = dict['image'].to(device)
            name = dict['name']
            w = dict['w']
            h = dict['h']

            out, _, _, _ = model(x)
            image = out.squeeze(0).cpu().numpy()
            image *= 255

            for index, _ in enumerate(name):
                resize = cv2.resize(image[index], (int(w[index]), int(h[index])))
                resize.astype('uint8')
                cv2.imwrite(data_root+"predict/"+name[index]+".png", resize)


            print("saved {}/{}".format((batch_idx+1)*len(x), len(test_loader.dataset)))
        end = time.process_time()
        print("done！ takes {} seconds".format(end-start))


test(net, test_data_loader, device)
