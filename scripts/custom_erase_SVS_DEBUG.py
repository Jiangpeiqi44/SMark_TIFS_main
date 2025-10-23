import os
import sys
sys.path.append(".")
sys.path.append("..")

import json
import logging
from random import shuffle
import random

from e4s.src.datasets.dataset import (
    CelebAHQDataset,
    get_transforms,
    TO_TENSOR,
    NORMALIZE,
    MASK_CONVERT_TF,
    FFHQDataset,
    FFHQ_MASK_CONVERT_TF,
    MASK_CONVERT_TF_DETAILED,
    FFHQ_MASK_CONVERT_TF_DETAILED,
)
from e4s.src.utils import torch_utils
from e4s.src.datasets.dataset import (
    CelebAHQDataset,
    get_transforms,
    TO_TENSOR,
    NORMALIZE,
    MASK_CONVERT_TF,
    FFHQDataset,
    FFHQ_MASK_CONVERT_TF,
    MASK_CONVERT_TF_DETAILED,
    FFHQ_MASK_CONVERT_TF_DETAILED,
)
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from options.options import EvalV8Options

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.signal import max_len_seq
from torch import nn

class InjectionDatasetREAD(Dataset):
    def __init__(self, root, num, DFtype, size=1024, label_transform=None, ds_type='CAHQ'):
        # 加载原始图像路径
        self.image_paths = sorted([x.path for x in os.scandir(root) if x.name.endswith(".jpg") or x.name.endswith(".png")])[:num]
        # 加载深度伪造图像路径
        self.image_df_paths = [i.replace('WM', DFtype) for i in self.image_paths]
        # 分割结果的根目录

        self.seg_root = '/home/gdata/face/e4s/FFHQ/BiSeNet_mask'

        self.ds_type = ds_type
        if self.ds_type == 'CAHQ':
            self.seg_root = '/home/gdata/face/e4s/CelebAMask-HQ/CelebA-HQ-mask'
        print('[*]{} images have been loaded'.format(len(self.image_paths)))
        print('[*]{} DF images have been loaded'.format(len(self.image_df_paths)))

        # 图像预处理
        self.transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        # 标签预处理
        self.label_transform = label_transform

    def __len__(self):
        return len(self.image_paths)

    def get_segmentation_path(self, image_name):
        """
        根据图像名称推算分割结果的路径。
        """
        # 提取图像编号
        base_name = os.path.basename(image_name)  # 提取文件名
        number = int(base_name.split('.')[0])  # 提取编号（假设文件名形如 "00001.PNG"）
        folder_name = f"{(number // 1000)*1000:05d}"  # 每1000个文件分一个文件夹
        if self.ds_type == 'CAHQ':
            folder_name=''
        return os.path.join(self.seg_root, folder_name, base_name)

    def __getitem__(self, index):
        # 加载原始图像
        image_path = self.image_paths[index]
        base_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path).convert('RGB')

        # 加载深度伪造图像
        image_df_path = self.image_df_paths[index]
        image_df = Image.open(image_df_path).convert('RGB')

        # 计算分割结果路径并加载
        seg_path = self.get_segmentation_path(image_path)
        seg = Image.open(seg_path).convert('L')  # 分割结果为灰度图

        # 应用预处理
        image = self.transforms(image)
        image_df = self.transforms(image_df)
        seg = self.label_transform(seg) if self.label_transform else self.transforms(seg)

        return image, image_df, seg, base_name
    
def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)
def starnorm(x):
    """Convert the range from [0, 1] to [-1, 1]."""
    out = (x - 0.5) * 2
    return out.clamp_(-1, 1)
if __name__ == "__main__":
    DFtype='HFGI' #'StarGAN' #'SimSwap'
    ds_type='CAHQ'
    dataset = InjectionDatasetREAD(root=f'./data/WM_{ds_type}', num=1000, DFtype=DFtype, size=1024, label_transform=transforms.Compose(
                    [FFHQ_MASK_CONVERT_TF_DETAILED, TO_TENSOR]), ds_type=ds_type)

    dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=int(4),
            drop_last=False,
        )
    for i, input_batch in enumerate(tqdm(dataloader)):
        img_rec, img_rec_df, mask, base_name = input_batch
        mask = (mask * 255).long()
                    # [bs,1,H,W] format mask to one-hot，i.e., [bs,#seg_cls,H,W]
        onehot = torch_utils.labelMap2OneHot(
            mask, num_cls=12
        )

        # 获取 batch size 和其他维度
        bs, num_cls, H, W = onehot.shape
        
        # 将 one-hot 掩码从 512x512 缩放到 1024x1024
        onehot_resized = F.interpolate(onehot.float(), size=(1024, 1024), mode='nearest')
        output_dir = "/home/jpq/StyleMark/data/erase_CAHQ"
        # 遍历每个样本，随机置零或随机化两个区域
        for b in range(bs):
            # 初始化掩码区域的权重
            mask_weights = torch.ones((1024, 1024), device=onehot.device)  # 默认全为 1
            
            # 随机选择两个区域索引
            # selected_classes = random.sample(range(num_cls), 2)
            selected_classes = [0, 6]
            
            for cls in selected_classes:
                # 生成随机区域的权重：0 或 [-1, 1] 的随机值
                random_mask = torch.rand(1024, 1024, device=onehot.device)# * 2 - 1  # 随机数范围为 [-1, 1]
                region_mask = onehot_resized[b, cls, :, :]
                
                # 根据区域掩码更新权重
                if random.random() < 0.5:  # 50% 概率置为全 0
                    mask_weights = mask_weights * (1 - region_mask)
                else:  # 50% 概率置为随机数
                    mask_weights = mask_weights * (1 - region_mask + region_mask * random_mask)

            # 覆盖 img_rec：元素级相乘
            modified_img = (img_rec[b]) * mask_weights.unsqueeze(0)  # [3, 1024, 1024]

            # 保存修改后的图像
            save_name = f"{base_name[b]}.png"  # 假设 base_name 中包含文件名（无扩展名）
            save_path = os.path.join(output_dir, save_name)
            
            # 转换为 NumPy 格式并保存
            modified_img_np = (modified_img * 255).byte().permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            Image.fromarray(modified_img_np).save(save_path)

            # print(f"Saved to {save_path}")