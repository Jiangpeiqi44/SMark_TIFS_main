import os
import sys
import json
import logging
from random import shuffle

sys.path.append(".")
sys.path.append("..")

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.signal import max_len_seq
from torch import nn
import re
# from HFGI.configs import data_configs, paths_config
# from HFGI.utils.model_utils import setup_model
# from HFGI.utils.common import tensor2im
# from HFGI.editings import latent_editor
import torchvision.transforms as transforms
# from DiffFace.optimization.image_editor import ImageEditor
from torch.utils.data import Dataset
# from DiffFace.optimization.arguments import get_arguments

import torch
from torch.utils.data import DataLoader

from options.options import EvalV8Options
import matplotlib.pyplot as plt
import cv2
import math
import logging
import scripts.distortion as distortion

# from network.AAD_WM_v7_conj import AADGenerator
# from network.MAE import MLAttrEncoder
# from network.encoder import Encoder
# from network.decoder import Decoder
# from network.Encoder_U import DW_Encoder
# from network.Decoder_U import DW_Decoder
from face_modules.model import Backbone
from criteria import loss_functions
from utils.common import (
    l2_norm,
    alignment,
    tensor2img,
    generate_seqstate,
    calculatie_correlation,
    calculatie_correlation_multi_e4s,
    evaluation,
    calculatie_correlation_multi_wide,
)
from utils.dataset import InjectionDataset, InjectionDatasetWithStarGAN
from network.Random_Noise import Random_Noise
import argparse
from SimSwap.models.models import create_model_lite
from StarGAN.solver import Solver
from torch.cuda.amp import autocast as autocast, GradScaler
from e4s.src.models.networks import Net3
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
# 统一设置字体
from os.path import abspath, dirname, join

import numpy as np
import scipy.sparse as sp
from proplot import rc

rc["font.family"] = "TeX Gyre Schola"
FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, "data")

MACOSKO_COLORS = {
    "Amacrine cells": "#A5C93D",
    "Astrocytes": "#8B006B",
    "Bipolar cells": "#2000D7",
    "Cones": "#538CBA",
    "Fibroblasts": "#8B006B",
    "Horizontal cells": "#B33B19",
    "Microglia": "#8B006B",
    "Muller glia": "#8B006B",
    "Pericytes": "#8B006B",
    "Retinal ganglion cells": "#C38A1F",
    "Rods": "#538CBA",
    "Vascular endothelium": "#8B006B",
}
ZEISEL_COLORS = {
    "Astroependymal cells": "#d7abd4",
    "Cerebellum neurons": "#2d74bf",
    "Cholinergic, monoaminergic and peptidergic neurons": "#9e3d1b",
    "Di- and mesencephalon neurons": "#3b1b59",
    "Enteric neurons": "#1b5d2f",
    "Hindbrain neurons": "#51bc4c",
    "Immature neural": "#ffcb9a",
    "Immune cells": "#768281",
    "Neural crest-like glia": "#a0daaa",
    "Oligodendrocytes": "#8c7d2b",
    "Peripheral sensory neurons": "#98cc41",
    "Spinal cord neurons": "#c52d94",
    "Sympathetic neurons": "#11337d",
    "Telencephalon interneurons": "#ff9f2b",
    "Telencephalon projecting neurons": "#fea7c1",
    "Vascular cells": "#3d672d",
}
MOUSE_10X_COLORS = {
    0: "#FFFF00",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
    22: "#FF2F80",
    23: "#61615A",
    24: "#BA0900",
    25: "#6B7900",
    26: "#00C2A0",
    27: "#FFAA92",
    28: "#FF90C9",
    29: "#B903AA",
    30: "#D16100",
    31: "#DDEFFF",
    32: "#000035",
    33: "#7B4F4B",
    34: "#A1C299",
    35: "#300018",
    36: "#0AA6D8",
    37: "#013349",
    38: "#00846F",
}
# # background 0 , eyebrows 1 , eyes 2 , nose 3 , mouth 4 , lips 5 , face skin 6 , neck 7 , hair 8 , ears 9 , eyeglass 10 , and ear rings 11
C_list = range(10)
L_list = range(13)
factor = 0.5
WM_layer_list = []
for c in C_list:
    for l in L_list:
        WM_layer_list.append((c, l, factor))
(MASK_C_SELECT, GAN_LAYER_SELECT, _) = (6, 9, 0)


def ahead_one(a):
    b = a.pop(0)
    a.append(b)
    return a


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def perturbation(im, mode="None"):
    if mode == "None":
        return im
    type, level = mode.split("#")
    level = float(level)
    im = im.resize((256, 256))
    im = np.asarray(im)
    im = np.flip(im, 2)
    if type == "CS":
        im = distortion.color_saturation(im, level)
    elif type == "CC":
        im = distortion.color_contrast(im, level)
    elif type == "BW":
        im = distortion.block_wise(im, int(level))
    elif type == "GNC":
        im = distortion.gaussian_noise_color(im, level)
    elif type == "GB":
        im = distortion.gaussian_blur(im, int(level))
    elif type == "JPEG":
        im = distortion.jpeg_compression(im, int(level))
    elif type == "VC":
        im = distortion.video_compression(im, int(level))
    elif type == "REAL_JPEG":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(level)]
        result, encimg = cv2.imencode(".jpg", im, encode_param)
        im = cv2.imdecode(encimg, 1)
    else:
        logging.info("Error mode:", mode)
        exit(0)
    im = Image.fromarray(np.flip(im, 2))
    return im


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def save_dict_to_json(data_dict, output_file):
    """
    将字典保存为 JSON 文件。

    Parameters:
        data_dict (dict): 要保存的字典。
        output_file (str): 保存文件的路径，例如 "data.json"。
    """
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
    print(f"JSON saved to {output_file}")

def load_json_to_dict(input_file):
    """
    从 JSON 文件中读取内容并加载为字典。

    Parameters:
        input_file (str): 要读取的 JSON 文件路径。

    Returns:
        dict: 从 JSON 文件中加载的字典。
    """
    with open(input_file, "r", encoding="utf-8") as json_file:
        data_dict = json.load(json_file)
    print(f"Load JSON from {input_file}")
    return data_dict

def cal_PAPR(corr_abs):
    corr_abs = np.abs(corr_abs)
    corr_seq_len = corr_abs.shape[0]
    assert corr_seq_len % 2 == 1
    peak = corr_abs[corr_seq_len // 2]
    corr_delpeak = np.delete(corr_abs, obj=corr_seq_len // 2, axis=None)
    avg = np.mean(corr_delpeak)
    return peak / (avg + 1e-7)

def plot_heatmap(data_dict, output_file):
    """
    绘制并保存二维热力图。
    
    Parameters:
        data_dict (dict): 包含二维数据的字典，格式为 {"C1_L1": value, "C1_L2": value, ...}。
        output_file (str): 保存图像的文件路径，例如 "heatmap.png"。
    """
    # 提取MASK_C和GAN_LAYER的范围
    keys = list(data_dict.keys())
    mask_c_set = sorted(set(int(k.split("_")[0][1:]) for k in keys))
    gan_layer_set = sorted(set(int(k.split("_")[1][1:]) for k in keys))
    
    # 创建二维矩阵
    heatmap = np.zeros((len(mask_c_set), len(gan_layer_set)))
    for key, value in data_dict.items():
        mask_c = int(key.split("_")[0][1:])
        gan_layer = int(key.split("_")[1][1:])
        heatmap[mask_c_set.index(mask_c), gan_layer_set.index(gan_layer)] = value
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap="viridis", origin="lower", aspect="auto")
    plt.colorbar(label="Value")
    
    # 设置坐标轴标签
    plt.xticks(ticks=range(len(gan_layer_set)), labels=[f"L{l}" for l in gan_layer_set])
    plt.yticks(ticks=range(len(mask_c_set)), labels=[f"C{c}" for c in mask_c_set])
    plt.xlabel("GAN_LAYER")
    plt.ylabel("MASK_C")
    plt.title("2D Heatmap")
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved to {output_file}")

def parse_ranges(ext_papr_dict):
    """
    从 EXT_PAPR_dict 的键中解析出 MASK_C 和 GAN_LAYER 的范围。
    
    Parameters:
        ext_papr_dict (dict): 包含相关性数据的字典。
    
    Returns:
        tuple: (mask_c_range, gan_layer_range, inner_mask_c_range, inner_gan_layer_range)
    """
    mask_c_set, gan_layer_set = set(), set()
    inner_mask_c_set, inner_gan_layer_set = set(), set()
    
    pattern = r"C(\d+)_L(\d+)_OVR_C(\d+)_L(\d+)"
    
    for key in ext_papr_dict.keys():
        match = re.match(pattern, key)
        if match:
            mask_c, gan_layer, inner_mask_c, inner_gan_layer = map(int, match.groups())
            mask_c_set.add(mask_c)
            gan_layer_set.add(gan_layer)
            inner_mask_c_set.add(inner_mask_c)
            inner_gan_layer_set.add(inner_gan_layer)
    
    return (sorted(mask_c_set), sorted(gan_layer_set), sorted(inner_mask_c_set), sorted(inner_gan_layer_set))

def plot_extended_papr_heatmap(ext_papr_dict, output_file="heatmap.png"):
    """
    绘制热力图，每一行代表一个 MASK_C 和 GAN_LAYER 对其他组合的相关性。
    
    Parameters:
        ext_papr_dict (dict): 包含相关性数据的字典。
        output_file (str): 热力图保存路径。
    """
    # 自动解析范围
    mask_c_range, gan_layer_range, inner_mask_c_range, inner_gan_layer_range = parse_ranges(ext_papr_dict)
    
    # 计算外层行数和内层列数
    outer_size = len(mask_c_range) * len(gan_layer_range)
    inner_size = len(inner_mask_c_range) * len(inner_gan_layer_range)
    
    # 初始化热力图矩阵
    heatmap = np.zeros((outer_size, inner_size))
    row_labels = [f"C{m}_L{g}" for m in mask_c_range for g in gan_layer_range]
    col_labels = [f"C{im}_L{ig}" for im in inner_mask_c_range for ig in inner_gan_layer_range]

    # 填充热力图数据
    for row_idx, (mask_c, gan_layer) in enumerate([(m, g) for m in mask_c_range for g in gan_layer_range]):
        for col_idx, (inner_mask_c, inner_gan_layer) in enumerate([(im, ig) for im in inner_mask_c_range for ig in inner_gan_layer_range]):
            key = f"C{mask_c}_L{gan_layer}_OVR_C{inner_mask_c}_L{inner_gan_layer}"
            heatmap[row_idx, col_idx] = ext_papr_dict.get(key, 0)  # 默认值为 0

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap="viridis", aspect="auto")
    plt.colorbar(label="Correlation")
    
    # 设置轴标签
    plt.xticks(ticks=np.arange(len(col_labels)), labels=col_labels, rotation=90, fontsize=6)
    plt.yticks(ticks=np.arange(len(row_labels)), labels=row_labels, fontsize=6)
    plt.xlabel("Inner_MASK_C and Inner_GAN_LAYER")
    plt.ylabel("Outer_MASK_C and GAN_LAYER")
    plt.title("Extended PAPR Heatmap")
    
    # 保存图像
    plt.tight_layout(pad=0.5)
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved to {output_file}")


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

    
class HiDDenConfiguration:
    """
    The HiDDeN network configuration.
    """

    def __init__(
        self,
        H: int,
        W: int,
        message_length: int,
        encoder_blocks: int,
        encoder_channels: int,
        decoder_blocks: int,
        decoder_channels: int,
        use_discriminator: bool,
        use_vgg: bool,
        discriminator_blocks: int,
        discriminator_channels: int,
        decoder_loss: float,
        encoder_loss: float,
        adversarial_loss: float,
        enable_fp16: bool = False,
    ):
        self.H = H
        self.W = W
        self.message_length = message_length
        self.encoder_blocks = encoder_blocks
        self.encoder_channels = encoder_channels
        self.use_discriminator = use_discriminator
        self.use_vgg = use_vgg
        self.decoder_blocks = decoder_blocks
        self.decoder_channels = decoder_channels
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.decoder_loss = decoder_loss
        self.encoder_loss = encoder_loss
        self.adversarial_loss = adversarial_loss
        self.enable_fp16 = enable_fp16


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    # print(codes.shape)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, is_cars=False):
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            x, _, __ = batch
            inputs = x.to(
                torch.device("cuda" if torch.cuda.is_available else "cpu")
            ).float()
            latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
            i += len(latents)
    return torch.cat(all_latents)


class Inject:
    def __init__(self, opts, noise_config, star_config, DF_model, star_model, args_e4s):
        self.opts = opts
        self.noise_config = noise_config
        self.star_config = star_config
        self.DF_model = DF_model  # 已经在cuda:0上了
        self.DF_model.eval()
        self.star_model = star_model  # 已经在cuda:0上了

        self.global_step = 0

        self.seqs_dict = load_json_to_dict('./data/seqs_dict.json')
        torch.backends.deterministic = True
        SEED = self.opts.seed
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        self.opts.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        logging.info("[*]Running on device: {}".format(self.opts.device))

        # args_diff = argparse.ArgumentParser()
        # args_dict_diff = vars(args_diff)
        # with open('diffface.json', 'rt') as f:
        #     args_dict_diff.update(json.load(f))
        # self.image_editor = ImageEditor(args_diff)

        # ckpt='HFGI/ckpt.pt'
        # self.net_hfgi, self.opts_hfgi = setup_model(ckpt, self.opts.device)

        self.noiser = Random_Noise(noise_config)

        logging.info(
            "[*]Loading Face Recognition Model {} from {}".format(
                self.opts.facenet_mode, self.opts.facenet_dir
            )
        )
        if self.opts.facenet_mode == "arcface":
            self.facenet = Backbone(
                input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
            ).to(self.opts.device)
            self.facenet.load_state_dict(
                torch.load(
                    os.path.join(self.opts.facenet_dir, "model_ir_se50.pth"),
                    map_location=self.opts.device,
                ),
                strict=True,
            )
        elif self.opts.facenet_mode == "circularface":
            self.facenet = Backbone(
                input_size=112, num_layers=100, drop_ratio=0.4, mode="ir", affine=False
            ).to(self.opts.device)
            self.facenet.load_state_dict(
                torch.load(
                    os.path.join(self.opts.facenet_dir, "CurricularFace_Backbone.pth"),
                    map_location=self.opts.device,
                ),
                strict=True,
            )
        else:
            raise ValueError(
                "Invalid Face Recognition Model. Must be one of [arcface, CurricularFace]"
            )
        self.facenet.eval()

        self.msg_input = []
        self.mls_input = []
        self.rec_loss = loss_functions.RecLoss(
            self.opts.rec_weight, self.opts.recloss_mode, self.opts.device
        )
        self.e4s_config = args_e4s
        self.e4s_config.device = self.opts.device
        """E4SNet"""
        # TODO E4S model
        self.net = Net3(self.e4s_config)
        self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net = self.net.to(self.opts.device)
        save_dict = torch.load(self.e4s_config.checkpoint_path, map_location="cpu")
        # Net3-PSP权重
        # E4S Net3权重: 包含PSP+G+MLPs
        if True:
            self.net.load_state_dict(torch.load(self.opts.Net3_dir))
            self.net.latent_avg = save_dict["latent_avg"].to(opts.device)
            logging.info("Load pre-trained Net3 model!")
            logging.info(f"Load latent_avg success! size: {self.net.latent_avg.shape}")
        else:
            if self.opts.local_rank == 0:
                logging.info("Load scratch Net3!")
        if True:
            self.net.latent_avg = None
        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.latent_avg is None:
            self.net.latent_avg = self.net.G.mean_latent(int(1e5))[0].detach()
            if self.e4s_config.learn_in_w:
                self.net.latent_avg = self.net.latent_avg.repeat(1, 1)
            else:
                self.net.latent_avg = self.net.latent_avg.repeat(
                    2 * int(math.log(self.e4s_config.out_size, 2)) - 2, 1
                )
        self.log_size = int(math.log(self.e4s_config.out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        if True:
            self.noise_zero = None
        else:
            self.noise_zero = []
            for layer_idx in range(self.num_layers):
                res = (layer_idx + 5) // 2
                shape = [1, 1, 2**res, 2**res]
                self.noise_zero.append(torch.zeros(*shape).to(self.opts.device))
      

        self.seq_path = os.path.join(self.opts.output_dir, "sequence.txt")
        self.control_seq_path = os.path.join(
            self.opts.output_dir, "control_sequence.txt"
        )
        logging.info(
            "[*]Outputing {} sequence to {}".format(self.opts.seq_type, self.seq_path)
        )
        # np.savetxt(self.seq_path, seq, fmt='%f', delimiter=',')
        # if self.opts.seq_type == 'mls':
        #     self.state = state
        #     np.savetxt(self.control_seq_path, state, fmt='%f', delimiter=',')
        # self.seq = torch.from_numpy(seq).float().to(self.opts.device)

        # _, __, test_list = self.split_dataset(path=self.opts.img_dir, val_ratio=0.1, test_ratio=0.1)

        # self.dataset = InjectionDatasetWithStarGAN(root=test_list, attr_path=self.star_config.attr_path, selected_attrs=self.star_config.selected_attrs, max_num=self.opts.max_num, size=self.opts.size)
        # TODO load custom datasets
        DFtype = 'erase'  #'erase' #'StarGAN' #'SimSwap' 'HFGI'
        ds_type = 'CAHQ'
        self.dataset = InjectionDatasetREAD(root=f'./data/WM_{ds_type}', num=self.opts.max_num, DFtype=DFtype, size=self.opts.size, label_transform=transforms.Compose(
                    [FFHQ_MASK_CONVERT_TF_DETAILED, TO_TENSOR]), ds_type=ds_type)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.opts.batch_size,
            shuffle=False,
            num_workers=int(self.opts.num_workers),
            drop_last=False,
        )
    def cal_PAPR(self, corr_abs):
        """计算峰值到平均功率比(Peak-to-Average Power Ratio)"""
        corr_abs = np.abs(corr_abs)
        corr_seq_len = corr_abs.shape[0]
        assert corr_seq_len % 2 == 1, "相关序列长度应为奇数"
        peak = corr_abs[corr_seq_len // 2]
        corr_delpeak = np.delete(corr_abs, obj=corr_seq_len // 2, axis=None)
        avg = np.mean(corr_delpeak)
        return peak / (avg + 1e-7)

    def evaluate_thresholds(self, output_dir, min_threshold=0.5, max_threshold=20.0, step=0.5):
        """
        评估不同阈值下的检测准确率，并保存结果
        
        参数:
            output_dir: 结果保存目录
            min_threshold: 最小阈值
            max_threshold: 最大阈值
            step: 阈值步长
        """
        import os
        import numpy as np
        import json
        import matplotlib.pyplot as plt
        from sklearn.metrics import accuracy_score
        import copy
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Evaluating thresholds from {min_threshold} to {max_threshold} with step {step}")
        
        # 保存原始状态，避免影响主流程
        original_total_num = copy.deepcopy(self.total_num)
        original_global_step = copy.deepcopy(self.global_step)
        
        # 存储所有图像的PAPR值和真实标签
        all_papr_values = []
        all_labels = []
        
        # 遍历所有批次，收集PAPR值和真实标签
        for input_batch in tqdm(self.dataloader, desc="Collecting PAPR values"):
            with torch.no_grad():
                img_rec, img_rec_df, mask, base_name = input_batch
                img_rec = img_rec.to(self.opts.device, non_blocking=True).float()
                img_rec_df = img_rec_df.to(self.opts.device, non_blocking=True).float()
                
                # 正确转换mask为one-hot格式
                mask = (mask * 255).long().to(self.opts.device, non_blocking=True)
                onehot = torch_utils.labelMap2OneHot(mask, num_cls=self.e4s_config.num_seg_cls)
                
                # 获取水印序列
                seqs = []
                base_name_list = []
                for one_base_name in base_name:
                    if torch.is_tensor(one_base_name):
                        one_base_name = one_base_name.item()
                    one_base_name = str(one_base_name)
                    
                    # 确保one_base_name在seqs_dict中
                    if one_base_name in self.seqs_dict:
                        seqs.append(self.seqs_dict[one_base_name])
                        base_name_list.append(one_base_name)
                    else:
                        print(f"Warning: {one_base_name} not found in seqs_dict")
                
                # 如果没有有效序列，跳过这个批次
                if not seqs:
                    continue
                    
                seqs = np.array(seqs)
                
                # ============== 关键修改：分别处理真实和伪造图像 ==============
                # 处理真实水印图像 (img_rec)
                style_vectors_orig, _ = self.net.get_style_vectors(img_rec, onehot)
                style_codes_orig = self.net.cal_style_codes(style_vectors_orig)
                
                for i in range(len(img_rec)):
                    current_style_codes = style_codes_orig[i].cpu().numpy()
                    current_seqs = seqs[i]
                    
                    # 计算平均PAPR（只使用前12层）
                    total_papr = 0
                    count = 0
                    
                    for c in range(current_style_codes.shape[0]):
                        for l1 in range(min(12, current_style_codes.shape[1])):
                            for l2 in range(min(12, current_style_codes.shape[1])):
                                corr = np.correlate(
                                    current_style_codes[c, l1, :], 
                                    current_seqs,
                                    mode='full'
                                )
                                papr = self.cal_PAPR(np.abs(corr))
                                total_papr += papr
                                count += 1
                    
                    avg_papr = total_papr / count if count > 0 else 0
                    
                    # 真实水印图像的标签是0
                    all_papr_values.append(avg_papr)
                    all_labels.append(0)  # 0表示真实图像
                
                # 处理Deepfake图像 (img_rec_df)
                style_vectors_fake, _ = self.net.get_style_vectors(img_rec_df, onehot)
                style_codes_fake = self.net.cal_style_codes(style_vectors_fake)
                
                for i in range(len(img_rec_df)):
                    current_style_codes = style_codes_fake[i].cpu().numpy()
                    current_seqs = seqs[i]
                    
                    # 计算平均PAPR（只使用前12层）
                    total_papr = 0
                    count = 0
                    
                    for c in range(current_style_codes.shape[0]):
                        for l1 in range(min(12, current_style_codes.shape[1])):
                            for l2 in range(min(12, current_style_codes.shape[1])):
                                corr = np.correlate(
                                    current_style_codes[c, l1, :], 
                                    current_seqs,
                                    mode='full'
                                )
                                papr = self.cal_PAPR(np.abs(corr))
                                total_papr += papr
                                count += 1
                    
                    avg_papr = total_papr / count if count > 0 else 0
                    
                    # Deepfake图像的标签是1
                    all_papr_values.append(avg_papr)
                    all_labels.append(1)  # 1表示Deepfake图像
                # ============== 修改结束 ==============
        
        # 检查是否有数据
        if len(all_papr_values) == 0:
            print("No data collected for threshold evaluation")
            # 恢复原始状态
            self.total_num = original_total_num
            self.global_step = original_global_step
            return
        
        # 准备阈值列表
        thresholds = np.arange(min_threshold, max_threshold + step, step)
        accuracies = []
        
        # 为每个阈值计算准确率
        for threshold in thresholds:
            # 当PAPR < threshold 时，判定为Deepfake (标签1)
            # 当PAPR >= threshold 时，判定为真实 (标签0)
            predictions = (np.array(all_papr_values) < threshold).astype(int)
            
            # 计算准确率
            acc = accuracy_score(all_labels, predictions)
            accuracies.append(acc)
        
        # 转换为numpy数组
        accuracies = np.array(accuracies)
        
        # 找到最佳阈值
        best_idx = np.argmax(accuracies)
        best_threshold = thresholds[best_idx]
        best_accuracy = accuracies[best_idx]
        
        print(f"Best threshold: {best_threshold:.2f}, Accuracy: {best_accuracy:.4f}")
        
        # 保存结果到JSON
        results = {
            "thresholds": thresholds.tolist(),
            "accuracies": accuracies.tolist(),
            "best_threshold": float(best_threshold),
            "best_accuracy": float(best_accuracy),
            "total_samples": len(all_labels),
            "real_samples": int(np.sum(1 - np.array(all_labels))),
            "fake_samples": int(np.sum(all_labels))
        }
        
        json_path = os.path.join(output_dir, "threshold_evaluation.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {json_path}")
        
        # 绘制准确率随阈值变化的曲线
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, 'b-', linewidth=2)
        plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
        plt.xlabel('Threshold (τ)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy vs. Threshold', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图像
        img_path = os.path.join(output_dir, "threshold_accuracy_curve.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Accuracy curve saved to {img_path}")
        
        # 保存数据点，便于后续与其他方法合并比较
        data_points = {
            "x": thresholds.tolist(),
            "y": accuracies.tolist(),
            "label": "Current Method"
        }
        points_path = os.path.join(output_dir, "threshold_data_points.json")
        with open(points_path, "w") as f:
            json.dump(data_points, f, indent=4)
        
        print(f"Data points saved to {points_path}")
        
        # 恢复原始状态
        self.total_num = original_total_num
        self.global_step = original_global_step

    def deepfake(self, img_id, img_att):
        with torch.no_grad():
            latend_id = self.facenet(
                torch.nn.functional.interpolate(
                    img_id, size=(112, 112), mode="bilinear", align_corners=True
                )
            )
        latend_id = latend_id.detach()
        latend_id = latend_id / torch.norm(latend_id, p=2, dim=1, keepdim=True)
        ############## Forward Pass ######################
        img_fake = self.DF_model(img_att, latend_id, True)
        return img_fake

    def split_dataset(self, path, val_ratio=0.2, test_ratio=0.2, shuffle_flag=False):
        images = [
            x.path
            for x in os.scandir(path)
            if x.name.endswith(".jpg") or x.name.endswith(".png")
        ]
        if shuffle_flag:
            shuffle(images)
        total_len = len(images)
        train_images = images[: int(total_len * (1 - val_ratio - test_ratio))]
        val_images = images[
            int(total_len * (1 - val_ratio - test_ratio)) : int(
                total_len * (1 - test_ratio)
            )
        ]
        test_images = images[int(total_len * (1 - test_ratio)) :]
        return train_images, val_images, test_images

    # E4S ds
    def configure_datasets(self):
        if self.opts.dataset_name == "ffhq":
            train_ds = FFHQDataset(
                dataset_root=self.opts.ffhq_dataset_root,
                img_transform=transforms.Compose([TO_TENSOR, NORMALIZE]),
                label_transform=transforms.Compose(
                    [FFHQ_MASK_CONVERT_TF_DETAILED, TO_TENSOR]
                ),  # FFHQ_MASK_CONVERT_TF
                fraction=self.e4s_config.ds_frac,
                flip_p=self.e4s_config.flip_p,
                mode="train",
            )
        else:
            train_ds = CelebAHQDataset(
                dataset_root=self.opts.celeba_dataset_root,
                mode="train",
                img_transform=transforms.Compose([TO_TENSOR, NORMALIZE]),
                label_transform=transforms.Compose(
                    [MASK_CONVERT_TF_DETAILED, TO_TENSOR]
                ),  # MASK_CONVERT_TF_DETAILED
                fraction=self.e4s_config.ds_frac,
                flip_p=self.e4s_config.flip_p,
            )
        if self.opts.dataset_name == "ffhq":
            test_ds = FFHQDataset(
                dataset_root=self.opts.ffhq_dataset_root,
                img_transform=transforms.Compose([TO_TENSOR, NORMALIZE]),
                label_transform=transforms.Compose(
                    [FFHQ_MASK_CONVERT_TF_DETAILED, TO_TENSOR]
                ),  # FFHQ_MASK_CONVERT_TF
                fraction=self.e4s_config.ds_frac,
                flip_p=self.e4s_config.flip_p,
                mode="test",
            )
        else:
            test_ds = CelebAHQDataset(
                dataset_root=self.opts.celeba_dataset_root,
                mode="test",
                img_transform=transforms.Compose([TO_TENSOR, NORMALIZE]),
                label_transform=transforms.Compose(
                    [MASK_CONVERT_TF_DETAILED, TO_TENSOR]
                ),  # MASK_CONVERT_TF
                fraction=self.e4s_config.ds_frac,
            )

        return train_ds, test_ds

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def starnorm(self, x):
        """Convert the range from [0, 1] to [-1, 1]."""
        out = (x - 0.5) * 2
        return out.clamp_(-1, 1)

    def running(self):

        imgin_dir = os.path.join(self.opts.output_dir, "original_CAHQ")
        imgout_dir = os.path.join(self.opts.output_dir, "reconstructed_CAHQ")
        delta_dir = os.path.join(self.opts.output_dir, "delta_CAHQ")
        imgoutDF_dir = os.path.join(self.opts.output_dir, "reconstructed_DF_CAHQ")
        os.makedirs(imgin_dir, exist_ok=True)
        os.makedirs(imgout_dir, exist_ok=True)
        os.makedirs(delta_dir, exist_ok=True)
        os.makedirs(imgoutDF_dir, exist_ok=True)
        for per_type in range(len(self.noise_config)):
            psnr_sum = 0
            ssim_sum = 0
            lpips_sum = 0
            label_all = []
            pred_all = []
            pred_self_corr_all = []
            pred_semi_corr_all = []
            BER_list = []
            BER_semi_list = []
            logging.info(f"---[{self.noise_config[per_type]}]---")
            # self.seq_dict = {}
            '''Cal PAPR'''
            self.total_num = 0
            self.PAPR_dict = {}
            self.PAPR_DF_dict = {}

            self.EXT_PAPR_dict = {}
            self.EXT_PAPR_DF_dict = {}

            for MASK_C, GAN_LAYER, SEQ_GAMMA in WM_layer_list:
                self.PAPR_dict[f"C{MASK_C}_L{GAN_LAYER}"] = 0
                self.PAPR_DF_dict[f"C{MASK_C}_L{GAN_LAYER}"] = 0
                for Inner_MASK_C, Inner_GAN_LAYER, _ in WM_layer_list:
                    self.EXT_PAPR_dict[f"C{MASK_C}_L{GAN_LAYER}_OVR_C{Inner_MASK_C}_L{Inner_GAN_LAYER}"] = 0
                    self.EXT_PAPR_DF_dict[f"C{MASK_C}_L{GAN_LAYER}_OVR_C{Inner_MASK_C}_L{Inner_GAN_LAYER}"] = 0
            for i, input_batch in enumerate(tqdm(self.dataloader)):
                with torch.no_grad():
                    img_rec, img_rec_df, mask, base_name = input_batch
                    img_rec = img_rec.to(self.opts.device, non_blocking=True).float()
                    img_rec_df = img_rec_df.to(self.opts.device, non_blocking=True).float()
                    mask = (mask * 255).long().to(self.opts.device, non_blocking=True)
                    # [bs,1,H,W] format mask to one-hot，i.e., [bs,#seg_cls,H,W]
                    onehot = torch_utils.labelMap2OneHot(
                        mask, num_cls=self.e4s_config.num_seg_cls
                    )
                    batch_size = img_rec.shape[0]
                    seqs = []
                    for one_base_name in base_name:
                        seqs.append(self.seqs_dict[one_base_name])
                    seqs = np.array(seqs)
                    self.seq_len = 512

                    self.net.eval()
                    
                    """开始Evaluate"""

            # 在处理完所有数据后，评估阈值
            threshold_eval_dir = os.path.join('tao_HFGI/', "threshold_evaluation")
            self.evaluate_thresholds(threshold_eval_dir, min_threshold=0.1, max_threshold=25.0, step=0.2)
            delta_PAPR_dict = {}
            for MASK_C, GAN_LAYER, SEQ_GAMMA in WM_layer_list:
                self.PAPR_dict[f"C{MASK_C}_L{GAN_LAYER}"] = self.PAPR_dict[f"C{MASK_C}_L{GAN_LAYER}"]/self.total_num
                self.PAPR_DF_dict[f"C{MASK_C}_L{GAN_LAYER}"] = self.PAPR_DF_dict[f"C{MASK_C}_L{GAN_LAYER}"]/self.total_num
                delta_PAPR_dict[f"C{MASK_C}_L{GAN_LAYER}"] = self.PAPR_dict[f"C{MASK_C}_L{GAN_LAYER}"]-self.PAPR_DF_dict[f"C{MASK_C}_L{GAN_LAYER}"]
            delta_EXT_PAPR_dict = {}
            for MASK_C, GAN_LAYER, SEQ_GAMMA in WM_layer_list:
                for Inner_MASK_C, Inner_GAN_LAYER, _ in WM_layer_list:
                    self.EXT_PAPR_dict[f"C{MASK_C}_L{GAN_LAYER}_OVR_C{Inner_MASK_C}_L{Inner_GAN_LAYER}"] = self.EXT_PAPR_dict[f"C{MASK_C}_L{GAN_LAYER}_OVR_C{Inner_MASK_C}_L{Inner_GAN_LAYER}"]/self.total_num
                    self.EXT_PAPR_DF_dict[f"C{MASK_C}_L{GAN_LAYER}_OVR_C{Inner_MASK_C}_L{Inner_GAN_LAYER}"] = self.EXT_PAPR_DF_dict[f"C{MASK_C}_L{GAN_LAYER}_OVR_C{Inner_MASK_C}_L{Inner_GAN_LAYER}"]/self.total_num
                    delta_EXT_PAPR_dict[f"C{MASK_C}_L{GAN_LAYER}_OVR_C{Inner_MASK_C}_L{Inner_GAN_LAYER}"] = self.EXT_PAPR_dict[f"C{MASK_C}_L{GAN_LAYER}_OVR_C{Inner_MASK_C}_L{Inner_GAN_LAYER}"] - self.EXT_PAPR_DF_dict[f"C{MASK_C}_L{GAN_LAYER}_OVR_C{Inner_MASK_C}_L{Inner_GAN_LAYER}"]
            # logging.info(self.PAPR_dict)
            # logging.info(self.PAPR_DF_dict)
            # logging.info(delta_PAPR_dict)
            plot_heatmap(self.PAPR_dict, "WM_PAPR_heatmap.png")
            plot_heatmap(self.PAPR_DF_dict, "DF_PAPR_heatmap.png")
            plot_heatmap(delta_PAPR_dict, "delta_PAPR_heatmap.png")

            plot_extended_papr_heatmap(self.EXT_PAPR_dict,"WM_EXT_PAPR_heatmap.png")
            plot_extended_papr_heatmap(self.EXT_PAPR_DF_dict,"DF_EXT_PAPR_heatmap.png")
            plot_extended_papr_heatmap(delta_EXT_PAPR_dict,"delta_EXT_PAPR_heatmap.png")

            avg_PSNR = psnr_sum / self.global_step
            avg_SSIM = ssim_sum / self.global_step
            avg_LPIPS = lpips_sum / self.global_step
            logging.info(
                "[*]avgPSNR: {:.4f}, avgSSIM: {:.4f}, avgLPIPS: {:.4f}".format(
                    avg_PSNR, avg_SSIM, avg_LPIPS
                )
            )

            accuracy, precision, recall, f1_score, tn, fp, fn, tp = evaluation(
                label_all, pred_all
            )
            logging.info(
                "[*]Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1_Score: {:.4f}".format(
                    accuracy, precision, recall, f1_score
                )
            )
            logging.info(
                "[*]True Negative: {}, False Positive: {}, False Negative: {}, True Positive: {}".format(
                    tn, fp, fn, tp
                )
            )
            



def main():
    opts = EvalV8Options().parse()
    """添加SimSwap的伪造模型"""
    args_df = argparse.ArgumentParser()
    args_dict = vars(args_df)
    with open("opt.json", "rt") as f:
        args_dict.update(json.load(f))

    """E4S"""
    args_e4s = argparse.ArgumentParser()
    args_e4s_dict = vars(args_e4s)
    with open("e4s_opts_evalV8.json", "rt") as f:
        args_e4s_dict.update(json.load(f))

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    DF_model = create_model_lite(args_df, device=device)

    """StarGAN"""
    args_df_star = argparse.ArgumentParser()
    args_dict_star = vars(args_df_star)
    with open("stargan.json", "rt") as f:
        args_dict_star.update(json.load(f))
    args_df_star.image_size = opts.size
    # args_df_star.batch_size = opts.size
    args_df_star.dataset = "CelebA"
    solver = Solver(args_df_star)

    output_dir = os.path.join(opts.exp_dir, opts.Net3_dir.split("/")[2])
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "ROBUST.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    os.makedirs(output_dir, exist_ok=True)
    logging.info("[*]Generating injection results at {}".format(output_dir))

    opts.output_dir = output_dir

    opts_dict = vars(opts)
    logging.info(opts_dict)
    with open(os.path.join(output_dir, "inject_opts.json"), "w") as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)
    # noise_config = ['Identity()','JpegTest()','Resize()','GaussianBlur()','MedianBlur()','Brightness()','Contrast()','Saturation()','Hue()','SaltPepper()','GaussianNoise()']
    # noise_config = ['SaltPepper()','GaussianNoise()']
    noise_config = ["Identity()"]
    inject = Inject(opts, noise_config, args_df_star, DF_model, solver, args_e4s)
    inject.running()


if __name__ == "__main__":
    main()
