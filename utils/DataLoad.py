import torch
import cv2
import os
import glob
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.png'))

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('img', 'label')

        # 读取训练图片和标签图片
        image = cv2.imread(image_path, 0)  # 读取原图
        image = cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # 归一化

        tg_label = cv2.imread(label_path, 0)  # 读取真值label
        tg_label = cv2.threshold(tg_label, 128, 255, type=cv2.THRESH_BINARY)[1] / 255  # 二值化+归一化
        bg_label = 1 - tg_label  # 背景label
        # reshape为训练所需格式
        image = image.reshape(1, image.shape[0], image.shape[1])
        tg_label = tg_label.reshape(1, tg_label.shape[0], tg_label.shape[1])
        bg_label = bg_label.reshape(1, bg_label.shape[0], bg_label.shape[1])

        return image, [bg_label, tg_label]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        # 返回训练集大小
        return len(self.imgs_path)
