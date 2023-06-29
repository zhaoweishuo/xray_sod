import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import torch
import random


class LoadTrainDataset(data.Dataset):
    def __init__(self, data_root):
        # 读取文件list
        self.leaf_root = data_root
        csv = pd.read_csv(self.leaf_root+"list.csv")
        self.path_list = csv.iloc[:, 0]  # image column
        self.label_list = csv.iloc[:, 1]  # label column

        self.resize = 224
        self.image_num = len(self.label_list)

    def __getitem__(self, item):

        seed = random.randint(1, 999)

        img = Image.open(self.leaf_root+'image/'+self.path_list[item])

        label = Image.open(self.leaf_root+'gt/'+self.label_list[item])

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        label_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        torch.manual_seed(seed)
        img = image_transform(img)

        torch.manual_seed(seed)
        label = label_transform(label)


        return {
                'image': img,
                'label': label,
                }

    def __len__(self):

        return self.image_num


class LoadTestDataset(data.Dataset):
    def __init__(self, data_root):

        self.leaf_root = data_root
        csv = pd.read_csv(self.leaf_root+"list.csv")
        self.path_list = csv.iloc[:, 0]
        self.resize = 224
        self.image_num = len(self.path_list)

    def __getitem__(self, item):

        img = Image.open(self.leaf_root+'image/'+self.path_list[item])

        size = img.size
        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = image_transform(img)


        return {
                'image': img,
                'name': self.path_list[item].split('.')[0],
                'w': str(size[0]),
                'h': str(size[1]),
                }

    def __len__(self):

        return self.image_num

