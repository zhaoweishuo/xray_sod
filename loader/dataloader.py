import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
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
        """建立字典映射"""
        # 1.从文件中读取一个数据（例如，使用numpy.fromfile、PIL.Image.open）。
        # 2.对数据进行预处理（如torchvision.Transform）。
        # 3.返回数据对（例如图像和标签）。
        seed = random.randint(1, 999)

        img = Image.open(self.leaf_root+'image/'+self.path_list[item])

        label = Image.open(self.leaf_root+'gt/'+self.label_list[item])

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        label_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # 将图像转为Tensor
        ])

        torch.manual_seed(seed)  # 固定随机数使图像与标签增强一致
        img = image_transform(img)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理

        torch.manual_seed(seed)
        label = label_transform(label)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理

        # img.show()  #  显示图片需要注释掉transforms.ToTensor()
        # label.show()
        # exit()

        return {
                'image': img,
                'label': label,
                }

    def __len__(self):
        """返回数据长度"""
        return self.image_num


class LoadTestDataset(data.Dataset):
    def __init__(self, data_root):
        # 读取文件list
        self.leaf_root = data_root
        csv = pd.read_csv(self.leaf_root+"list.csv")
        self.path_list = csv.iloc[:, 0]  # 第一列路径列
        self.resize = 224
        self.image_num = len(self.path_list)

    def __getitem__(self, item):
        """建立字典映射"""
        # 1.从文件中读取一个数据（例如，使用numpy.fromfile、PIL.Image.open）。
        # 2.对数据进行预处理（如torchvision.Transform）。
        # 3.返回数据对（例如图像和标签）。
        img = Image.open(self.leaf_root+'image/'+self.path_list[item])

        size = img.size
        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = image_transform(img)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理

        # 返回自定义的dictionary
        return {
                'image': img,
                'name': self.path_list[item].split('.')[0],
                'w': str(size[0]),
                'h': str(size[1]),
                }

    def __len__(self):
        """返回数据长度"""
        return self.image_num


class LoadValDataset(data.Dataset):
    def __init__(self, data_root):
        # 读取文件list
        self.leaf_root = data_root
        csv = pd.read_csv(self.leaf_root+"list.csv")
        self.path_list = csv.iloc[:, 0]  # image column
        self.label_list = csv.iloc[:, 1]  # label column

        self.resize = 224
        self.image_num = len(self.label_list)

    def __getitem__(self, item):
        """建立字典映射"""
        # 1.从文件中读取一个数据（例如，使用numpy.fromfile、PIL.Image.open）。
        # 2.对数据进行预处理（如torchvision.Transform）。
        # 3.返回数据对（例如图像和标签）。
        img = Image.open(self.leaf_root+'image/'+self.path_list[item])
        label = Image.open(self.leaf_root+'gt/'+self.label_list[item])

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        label_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),  # 将图像转为Tensor
        ])

        img = image_transform(img)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理
        label = label_transform(label)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理

        return {
                'image': img,
                'label': label,
                }

    def __len__(self):
        """返回数据长度"""
        return self.image_num
