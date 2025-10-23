import sys

sys.path.append(".")
sys.path.append("..")
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from torch.utils import data
from ranger import Ranger
import warnings
import math
# from torch.cuda.amp import autocast as autocast, GradScaler
from utils.common import (
    l2_norm,
    alignment,
    visualize_train_results,
    visualize_train_results_multi,
    visualize_train_results_multi_e4s,
    generate_seqstate,
    visualize_train_results_multi_lite,
    visualize_train_results_multi_lite2,
)
import datetime
import torchvision.transforms as transforms
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
from e4s.src.models.networks import Net3
from e4s.src.models.stylegan2.model import Generator, Discriminator
from e4s.src.training.ranger import Ranger
from e4s.src.criteria.style_loss import StyleLoss
from e4s.src.criteria.adv_loss import AdvDLoss, AdvGLoss, DR1Loss, GPathRegularizer
from e4s.src.criteria.lpips.lpips import LPIPS
from e4s.src.criteria.face_parsing.face_parsing_loss import FaceParsingLoss
from e4s.src.criteria.id_loss import IDLoss
from e4s.src.criteria.w_norm import WNormLoss
from criteria import loss_functions
import torch.nn.functional as F
from collections import OrderedDict
from options.options import TrainingOptions
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import torch
import traceback
import matplotlib.pyplot as plt
from random import shuffle
from scipy.signal import max_len_seq
import numpy as np
import argparse
import os
import sys
import time
import json
import pprint
import logging

sys.path.append(".")
sys.path.append("..")

'''支持任意层+任意强度的组合'''
# MASK_C = 1
# # background, eyebrows, eyes, nose, mouth, lips, face skin, neck, hair, ears, eyeglass, and ear rings
# GAN_LAYER = 11
CL_stats={}
def cal_PAPR(corr_abs):
    corr_abs = np.abs(corr_abs)
    corr_seq_len = corr_abs.shape[0]
    assert corr_seq_len%2==1
    peak = corr_abs[corr_seq_len//2]
    corr_delpeak = np.delete(corr_abs, obj=corr_seq_len//2, axis=None)
    avg = np.mean(corr_delpeak)
    return peak / (avg+1e-7)

WM_layer_list = [(6,11,0.75), (0,1,0.5)]
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


class CosineAnnealingLRWarmup(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=1.0e-8,
        last_epoch=-1,
        verbose=False,
        warmup_steps=2,
        warmup_start_lr=1.0e-5,
    ):
        super(CosineAnnealingLRWarmup, self).__init__(
            optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
        )
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        if warmup_steps > 0:
            self.base_warup_factors = [
                (base_lr / warmup_start_lr) ** (1.0 / self.warmup_steps)
                for base_lr in self.base_lrs
            ]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if hasattr(self, "warmup_steps"):
            if self.last_epoch < self.warmup_steps:
                return [
                    self.warmup_start_lr * (warmup_factor**self.last_epoch)
                    for warmup_factor in self.base_warup_factors
                ]
            else:
                return [
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (self.last_epoch - self.warmup_steps)
                            / (self.T_max - self.warmup_steps)
                        )
                    )
                    * 0.5
                    for base_lr in self.base_lrs
                ]
        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
                / 2
                for base_lr in self.base_lrs
            ]


# ahead one for SimSwap
def ahead_one(a):
    b = a.pop(0)
    a.append(b)
    return a


class Train:
    def __init__(self, opts, noise_config, args_df, args_e4s, device):
        self.opts = opts
        self.opts.device = device
        self.simswap_config = args_df
        self.e4s_config = args_e4s
        self.e4s_config.device = device
        """E4SNet"""
        # TODO E4S model
        self.net = Net3(self.e4s_config)
        self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net = self.net.to(self.opts.device)
        save_dict = torch.load(self.e4s_config.checkpoint_path, map_location="cpu")
        # Net3-PSP权重
        if False:
            # print(save_dict["state_dict"].keys())
            print(
                "PSPEncoder: ",
                self.net.encoder.load_state_dict(
                    torch_utils.remove_module_prefix(
                        torch_utils.filter_module_prefix(
                            save_dict["state_dict"], prefix="encoder"
                        ),
                        prefix="module.encoder.",
                    ),
                    strict=False,
                ),
            )
            if self.opts.local_rank == 0:
                logging.info("Load pre-trained Net3-PSP!")
        else:
            if self.opts.local_rank == 0:
                logging.info("Load scratch PSP!")
        # E4S Net3权重: 包含PSP+G+MLPs
        if True:
            self.net.load_state_dict(
                torch_utils.remove_module_prefix(
                    save_dict["state_dict"], prefix="module."
                )
            )
            self.net.latent_avg = save_dict['latent_avg'].to(opts.device)
            if self.opts.local_rank == 0:
                logging.info("Load pre-trained Net3 model!")
            if self.opts.local_rank == 0:
                logging.info(f"Load latent_avg success! size: {self.net.latent_avg.shape}")
        else:
            if self.opts.local_rank == 0:
                logging.info("Load scratch Net3!")

        if self.opts.distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[self.opts.local_rank],
                output_device=self.opts.local_rank,
            )
            self.net = self.net.module

        # TODO handel D
        if self.e4s_config.train_D:
            self.D = Discriminator(self.e4s_config.out_size).to(self.opts.device).eval()

        if self.opts.distributed:
            self.D = torch.nn.parallel.DistributedDataParallel(
                self.D,
                device_ids=[self.opts.local_rank],
                output_device=self.opts.local_rank,
            )
            self.D = self.D.module
        # TODO styleGAN2
        styleGAN2_ckpt = torch.load(
            self.e4s_config.stylegan_weights, map_location=torch.device("cpu")
        )
        # D的权重
        if True:
            if self.e4s_config.train_D:
                if self.e4s_config.out_size == 1024:
                    self.D.load_state_dict(styleGAN2_ckpt["d"], strict=False)
                # 1024 resolution
                else:
                    self.custom_load_D_state_dict(
                        self.D, styleGAN2_ckpt["d"]
                    )  # load partial D
            if self.opts.local_rank == 0:
                logging.info("Loading pretrained D!")
        else:
            if self.opts.local_rank == 0:
                logging.info("Load scratch D!")
        # G的权重
        if False:
            self.net.G.load_state_dict(styleGAN2_ckpt["g_ema"], strict=False)
            if self.opts.local_rank == 0:
                logging.info("Loading pretrained G!")
        else:
            if self.opts.local_rank == 0:
                logging.info("Load scratch G!")
        # avg latent code
        if False:
            self.net.latent_avg = styleGAN2_ckpt["latent_avg"].to(self.opts.device)

            if self.e4s_config.learn_in_w:
                self.net.latent_avg = self.net.latent_avg.repeat(1, 1)
            else:
                self.net.latent_avg = self.net.latent_avg.repeat(
                    2 * int(math.log(self.e4s_config.out_size, 2)) - 2, 1
                )
            if self.opts.local_rank == 0:
                logging.info(f"Load latent_avg success! size: {self.net.latent_avg.shape}")
        if False:
            self.net.latent_avg = None
        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.latent_avg is None:
            self.net.latent_avg = self.net.G.mean_latent(int(1e5))[0].detach()
            if self.opts.local_rank == 0:
                logging.info(f"Load latent_avg success! size: {self.net.latent_avg.shape}")
        torch.backends.deterministic = False
        SEED = self.opts.seed
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        logging.info("[*]Running on device: {}".format(self.opts.device))
        """Cos 衰减"""
        # self.lr_scheduler_Net = CosineAnnealingLRWarmup(
        #     self.opt_aad,
        #     T_max=self.opts.max_epoch,
        #     eta_min=self.opts.min_lr,
        #     last_epoch=-1,
        #     warmup_steps=self.opts.warm_step,
        #     warmup_start_lr=self.opts.start_lr,
        # )
        # self.lr_scheduler_D = CosineAnnealingLRWarmup(self.opt_att,
        #                                         T_max=self.opts.max_epoch,
        #                                         eta_min=self.opts.min_lr,
        #                                         last_epoch=-1,
        #                                         warmup_steps=self.opts.warm_step,
        #                                         warmup_start_lr=self.opts.start_lr)
        self.mse_loss = nn.MSELoss().to(self.opts.device).eval()
        if self.e4s_config.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type="alex").to(self.opts.device).eval()
        if self.e4s_config.id_lambda > 0:
            self.id_loss = IDLoss(self.e4s_config).to(self.opts.device).eval()
        if self.e4s_config.face_parsing_lambda > 0:
            self.face_parsing_loss = (
                FaceParsingLoss(self.e4s_config).to(self.opts.device).eval()
            )
        if self.e4s_config.w_norm_lambda > 0:
            self.w_norm_loss = WNormLoss(
                start_from_latent_avg=self.e4s_config.start_from_latent_avg
            )
        if self.e4s_config.style_lambda > 0:  # gram matrix loss
            self.style_loss = (
                StyleLoss(
                    distance="l2",
                    VGG16_ACTIVATIONS_LIST=[3, 8, 15, 22],
                    normalize=self.e4s_config.style_loss_norm == 1,
                    in_size=self.e4s_config.out_size,
                )
                .to(self.opts.device)
                .eval()
            )

        self.adv_d_loss = AdvDLoss().to(self.opts.device).eval()
        self.adv_g_loss = AdvGLoss().to(self.opts.device).eval()
        self.d_r1_reg_loss = DR1Loss().to(self.opts.device).eval()
        self.g_path_reg_loss = GPathRegularizer().to(self.opts.device).eval()

        # Initialize optimizer
        self.optimizer, self.optimizer_D = self.configure_optimizers()

        self.trainpics_dir = os.path.join(self.opts.output_dir, "TrainPics")
        self.valpics_dir = os.path.join(self.opts.output_dir, "ValidationPics")
        self.checkpoints_dir = os.path.join(self.opts.output_dir, "CheckPoints")
        self.best_checkpoints_dir = os.path.join(self.opts.output_dir, "BestResult")
        self.log_dir = os.path.join(self.opts.output_dir, "Logs")
        if self.opts.local_rank == 0:
            ### Initialize result directories and folders ###
            os.makedirs(self.trainpics_dir, exist_ok=True)
            os.makedirs(self.valpics_dir, exist_ok=True)
            os.makedirs(self.checkpoints_dir, exist_ok=True)
            os.makedirs(self.best_checkpoints_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)

        ### Initialize loss functions ###
        self.style_wm_loss = loss_functions.MsgLoss(
            self.opts.wm_weight, self.opts.msgloss_mode
        ).to(self.opts.device)

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
                
        # TODO load custom datasets
        self.train_dataset, self.test_dataset = self.configure_datasets()
        if self.opts.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, shuffle=True
            )
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.opts.batch_size,
                num_workers=int(self.opts.num_workers),
                drop_last=True,
                pin_memory=True,
                sampler=self.train_sampler,
            )
        else:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.opts.batch_size,
                shuffle=True,
                num_workers=int(self.opts.num_workers),
                drop_last=True,
                pin_memory=True,
            )
        # test set
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.opts.test_batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True,
        )

        ### Initialize logger ###
        if self.opts.local_rank == 0:
            self.logger = SummaryWriter(log_dir=self.log_dir)
        self.best_loss = None

    def custom_load_D_state_dict(self, module, state_dict):
        """Load partial StyleGAN discriminator weights
        Args:
            module (nn.Module): the module to be updated
            state_dict (): styleGAN weights, convs.0 corresponds to 1024 resolution
        """
        local_state = {k: v for k, v in module.named_parameters() if v is not None}

        #
        del local_state["convs.0.0.weight"]
        del local_state["convs.0.1.bias"]

        idx_gap = int(math.log(1024, 2)) - int(math.log(self.opts.out_size, 2))

        new_state_dict = OrderedDict()
        for name, param in local_state.items():
            if name[:5] == "convs":
                layer_idx = int(name[6])
                name_in_pretrained = name[:6] + str(layer_idx + idx_gap) + name[7:]
                new_state_dict[name] = state_dict[name_in_pretrained]
            else:
                new_state_dict[name] = state_dict[name]  # FC

        module.load_state_dict(new_state_dict, strict=False)

    def configure_optimizers(self):
        self.params = list(
            filter(lambda p: p.requires_grad, list(self.net.parameters()))
        )
        self.params_D = (
            list(filter(lambda p: p.requires_grad, list(self.D.parameters())))
            if self.e4s_config.train_D
            else None
        )
        if self.opts.local_rank == 0:
            logging.info('===Net Params===')
            for name, param in self.net.named_parameters():
                logging.info(f'{name}: {param.requires_grad}')
            logging.info('===D Params===')
            for name, param in self.D.named_parameters():
                logging.info(f'{name}: {param.requires_grad}')
        d_reg_ratio = (
            self.e4s_config.d_reg_every / (self.e4s_config.d_reg_every + 1)
            if self.e4s_config.d_reg_every > 0
            else 1
        )

        if self.opts.optim_name == "adam":
            optimizer = torch.optim.Adam(self.params, lr=self.opts.lr)
            optimizer_D = (
                torch.optim.Adam(self.params_D, lr=self.opts.lr * d_reg_ratio)
                if self.e4s_config.train_D
                else None
            )
        elif self.opts.optim_name == "adamW":
            optimizer = torch.optim.AdamW(
                self.params, lr=self.opts.lr, betas=(0.9, 0.999), weight_decay=1e-2
            )
            optimizer_D = (
                torch.optim.AdamW(
                    self.params_D,
                    lr=self.opts.lr * d_reg_ratio,
                    betas=(0.9, 0.999),
                    weight_decay=1e-2,
                )
                if self.e4s_config.train_D
                else None
            )
        else:
            optimizer = Ranger(self.params, lr=self.opts.lr)
            optimizer_D = (
                Ranger(self.params_D, lr=self.opts.lr * d_reg_ratio)
                if self.e4s_config.train_D
                else None
            )
        return optimizer, optimizer_D

    def starnorm(self, x):
        """Convert the range from [0, 1] to [-1, 1]."""
        out = (x - 0.5) * 2
        return out.clamp_(-1, 1)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

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
        if self.opts.local_rank == 0:
            logging.info(f"Number of training samples: {len(train_ds)}")
            logging.info(f"Number of test samples: {len(test_ds)}")
        return train_ds, test_ds

    def calc_loss(self, img, recon1, mask, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        if self.e4s_config.face_parsing_lambda > 0:
            loss_face_parsing_1, face_parsing_sim_improvement_1 = (
                self.face_parsing_loss(recon1, img)
            )

            loss_dict["loss_face_parsing"] = float(loss_face_parsing_1)
            loss_dict["face_parsing_improve"] = float(face_parsing_sim_improvement_1)
            loss += loss_face_parsing_1 * self.e4s_config.face_parsing_lambda

        if self.e4s_config.id_lambda > 0:
            loss_id_1, sim_improvement_1, id_logs_1 = self.id_loss(recon1, img)
            loss_dict["loss_id"] = float(loss_id_1)
            loss_dict["id_improve"] = float(sim_improvement_1)
            loss += loss_id_1 * self.e4s_config.id_lambda

        if self.e4s_config.l2_lambda > 0:
            loss_l2_1 = F.mse_loss(recon1, img)
            loss_dict["loss_l2"] = float(loss_l2_1)
            loss += loss_l2_1 * self.e4s_config.l2_lambda

        if self.e4s_config.lpips_lambda > 0:
            loss_lpips = 0
            for i in range(3):
                loss_lpips_1 = self.lpips_loss(
                    F.adaptive_avg_pool2d(recon1, (1024 // 2**i, 1024 // 2**i)),
                    F.adaptive_avg_pool2d(img, (1024 // 2**i, 1024 // 2**i)),
                )
                loss_lpips += loss_lpips_1
            loss_dict["loss_lpips"] = float(loss_lpips)
            loss += loss_lpips * self.e4s_config.lpips_lambda

        if self.e4s_config.w_norm_lambda > 0:
            loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
            loss_dict["loss_w_norm"] = float(loss_w_norm)
            loss += loss_w_norm * self.e4s_config.w_norm_lambda

        if self.e4s_config.style_lambda > 0:  # gram matrix loss
            loss_style_1 = self.style_loss(
                recon1, img, mask_x=(mask == 3).float(), mask_x_hat=(mask == 3).float()
            )
            loss_dict["loss_style"] = float(loss_style_1)
            loss += loss_style_1 * self.e4s_config.style_lambda
        loss_dict["loss"] = float(loss)

        return (
            (loss, loss_dict, id_logs_1)
            if self.e4s_config.id_lambda > 0
            else (loss, loss_dict, None)
        )
    def debug(self, epoch, dataloader):
        # for i in range(12):
        #     for j in range(13):
        #         CL_stats[f'C{i}_L{j}']=np.zeros(10)
        GAMMA_list = np.linspace(start = 0, stop = 2, num = 10)
        # latents_ori = {}
        latents_rec = {}
        for gamma in GAMMA_list:
            # latents_ori[f'GAMMA_{gamma}']=[]
            latents_rec[f'GAMMA_{gamma}']=[]
        for batch_idx, batch in enumerate(dataloader):
            img_org, mask, mask_vis = batch
            img_org = img_org.to(self.opts.device, non_blocking=True).float()
            mask = (mask * 255).long().to(self.opts.device, non_blocking=True)
            # [bs,1,H,W] format mask to one-hot，i.e., [bs,#seg_cls,H,W]
            onehot = torch_utils.labelMap2OneHot(
                mask, num_cls=self.e4s_config.num_seg_cls
            )
            batch_size = img_org.shape[0]
            random_states = [
                generate_seqstate(self.opts.message) for _ in range(batch_size)
            ]
            random_states = np.array(random_states)
            seqs = [
                np.insert(
                    max_len_seq(nbits=self.opts.message, state=random_state)[0] * 2.0
                    - 1.0,
                    -1,
                    0,
                )
                for random_state in random_states
            ]
            seqs = np.array(seqs)
            # 0716: only add to C0_L0-17
            seqs = (
                torch.tensor(seqs)
                .float()
                .to(self.opts.device, non_blocking=True)
                .view(batch_size, 512)
                # .repeat(1, self.e4s_config.remaining_layer_idx, 1)
            )  # [B,18,512]
            # seqs_pad = torch.zeros(
            #     (batch_size, 18 - self.e4s_config.remaining_layer_idx, 512)
            # ).to(self.opts.device)
            # seqs = torch.cat([seqs, seqs_pad], dim=1)
            self.seqs_CL = (
                torch.zeros_like(seqs)
                .view(batch_size, 1, 1, 512)
                .repeat(1, 12, 18, 1)
                .to(self.opts.device, non_blocking=True)
            )  # [B,12,18,512]
            self.seq_len = 512
            with torch.no_grad():
                """ Training Generator """
                self.D.eval()
                self.net.eval()
                vis_dict = {}
                style_vectors, structure_feats = self.net.get_style_vectors(
                    img_org, onehot
                )
                style_codes = self.net.cal_style_codes(
                    style_vectors)  # [B, 12, 18, 512]
                for idx, SEQ_GAMMA in enumerate(GAMMA_list):
                    self.seq_weighted = self.seqs_CL + seqs.view(batch_size, 1, 1, 512)*SEQ_GAMMA

                    style_codes_ADD = (
                        style_codes * self.opts.idvec_weight + self.seq_weighted
                    )
                    # for i in range(12):
                    #     for j in range(13):
                    #         for k in range(style_codes_ADD.shape[0]):
                    #             single_style = style_codes_ADD[k, i, j, :].detach().cpu().view(-1).numpy()
                    #             ADD_acorr = np.correlate(single_style, seqs[k].cpu().view(-1).numpy(), "full")
                    #             ADD_PAPR = cal_PAPR(ADD_acorr)
                    #             CL_stats[f'C{i}_L{j}'][idx]+=ADD_PAPR
                                
                                
                    img_rec, latent, _ = self.net.gen_img(
                        structure_feats,
                        style_codes_ADD,
                        onehot,
                        noise=self.noise_zero,
                        return_latents=True,
                    )
                    # g_loss = torch.tensor(0.0, device=self.opts.device)
                    # if self.e4s_config.train_D:
                    #     fake_pred_1 = self.D(img_rec)
                    #     g_loss = self.adv_g_loss(fake_pred_1)
                    """提取reconst的styleCode"""
                    # torch_utils.requires_grad(self.net, False)
                    style_vectors_rec, _ = self.net.get_style_vectors(img_rec, onehot)
                    style_codes_rec = self.net.cal_style_codes(style_vectors_rec)
                    latents_rec[f'GAMMA_{SEQ_GAMMA}'].append(style_codes_rec.detach().cpu().numpy().tolist())
                # torch_utils.requires_grad(self.net, True)
                # loss_, loss_dict, id_logs = self.calc_loss(img_org, img_rec, mask, latent)
                # loss_dict["g_loss"] = float(g_loss)

                # loss_wm_style = 0
                # for (MASK_C, GAN_LAYER, SEQ_GAMMA) in WM_layer_list:
                #     loss_wm_style += self.style_wm_loss(
                #     style_codes_ADD.float()[:, MASK_C, GAN_LAYER, :], style_codes_rec.float()[:, MASK_C, GAN_LAYER, :]
                # )
                # loss_dict["loss_wm_style"] = float(loss_wm_style/(self.opts.wm_weight*len(WM_layer_list)))

                # overall_loss = loss_ + self.e4s_config.g_adv_lambda * g_loss + loss_wm_style
                # loss_dict["loss"] = float(overall_loss)
                # # logging.info(overall_loss)
                # self.optimizer.zero_grad()
                # overall_loss.backward()
                # self.optimizer.step()
                # MASK_C, GAN_LAYER, _ = WM_layer_list[np.random.randint(0,len(WM_layer_list))]
                # vis_dict = {
                #     "img_org": self.denorm(img_org),
                #     "img_rec": self.denorm(img_rec),
                #     "img_df": self.denorm(img_rec),
                #     "id_org": style_codes[:, MASK_C, GAN_LAYER, :].view(batch_size, 512),
                #     "id_input": style_codes_ADD[:, MASK_C, GAN_LAYER, :].view(batch_size, 512),
                #     "id_rec": style_codes_rec[:, MASK_C, GAN_LAYER, :].view(batch_size, 512),
                #     "seq": seqs,
                #     "seq_len": self.seq_len,
                #     "seq_weighted": self.seq_weighted[:, MASK_C, GAN_LAYER, :].view(batch_size, 512),
                #     "MASK_C": MASK_C,
                #     "GAN_LAYER": GAN_LAYER,
                # }

                if (
                    self.opts.local_rank == 0
                    and (batch_idx + 1) % self.opts.board_interval == 0
                ):
                    self.print_metrics(CL_stats, batch_idx + 1, epoch, prefix="train")
                    # self.log_metrics(CL_stats, self.train_steps + 1, prefix="train")

                self.train_steps += 1

                if batch_idx == self.opts.max_train_iters - 1:
                    break
        if self.opts.local_rank == 0:
            with open(os.path.join(self.opts.output_dir, "latent_stats.json"), "w") as f:
                json.dump(latents_rec, f, indent=4, sort_keys=True)


            


    def debuging(self):
        logging.info("Start Debuging...")
        self.net.train()
        if self.e4s_config.train_D:
            self.D.train()
        self.train_steps = 0
        self.validate_steps = 0

        for epoch in range(self.opts.max_epoch):
            # torch.cuda.empty_cache()
            if self.opts.distributed:
                self.train_sampler.set_epoch(epoch)
            self.debug(epoch, self.train_dataloader)

        logging.info("Training Debug")

    def print_metrics(self, loss_dict, iteration, epoch, prefix):
        if prefix == "train":
            logging.info(
                "Metrics for train, iteration {:05d}, epoch {:04d}".format(
                    iteration, epoch
                )
            )
            for key, value in loss_dict.items():
                logging.info("\t{}: {}".format(key, value/(self.opts.batch_size*iteration)))
        elif prefix == "validate":
            logging.info(
                "Validate, iteration {:05d}, epoch {:04d} are".format(iteration, epoch)
            )
            logging.info(
                ["{}: {}".format(key, value) for key, value in loss_dict.items()]
            )
        else:
            raise ValueError("Unexpected prefix mode {}".format(prefix))

    def save_checkpoint(self, Dis, Net, epoch, is_best):
        if is_best:
            torch.save(
                Net.state_dict(),
                os.path.join(self.best_checkpoints_dir, "Net_best.pth"),
            )
            torch.save(
                Dis.state_dict(),
                os.path.join(self.best_checkpoints_dir, "Dis_best.pth"),
            )
        else:
            torch.save(
                Net.state_dict(),
                os.path.join(self.checkpoints_dir, "Net_{:05d}.pth".format(epoch)),
            )
            torch.save(
                Dis.state_dict(),
                os.path.join(self.checkpoints_dir, "Dis_{:05d}.pth".format(epoch)),
            )

    def log_metrics(self, loss_dict, iteration, prefix):
        for key, value in loss_dict.items():
            self.logger.add_scalar("{}/{}".format(prefix, key), value, (iteration))


def main():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    opts = TrainingOptions().parse()
    opts.local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opts.distributed = n_gpu > 1
    if opts.distributed:
        torch.cuda.set_device(opts.local_rank)
        device = torch.device("cuda", opts.local_rank)
        torch.distributed.init_process_group(backend="Gloo", init_method="env://")
        # synchronize()

    """E4S"""
    args_e4s = argparse.ArgumentParser()
    args_e4s_dict = vars(args_e4s)
    with open("e4s_opts.json", "rt") as f:
        args_e4s_dict.update(json.load(f))
    with open("e4s_opts.json", "rt") as f:
        args_e4s_json = json.load(f)
    # print(args_e4s_json)
    cur_time = time.strftime("%Y%m%d_H%H%M", time.localtime())
    output_dir = os.path.join(
        opts.exp_dir,
        "{}_{}".format(
            cur_time,
            opts.info,
        ),
    )
    if opts.local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(os.path.join(output_dir, "stdout.log")),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info("[*]Exporting experiment results at {}\n".format(output_dir))
    opts_dict = vars(opts)
    if opts.local_rank == 0:
        # pprint.pprint(opts_dict)
        logging.info("SepID Configuration:\n")
        logging.info(pprint.pformat(opts_dict))
    opts_dict_to_json = {}
    opts_dict_to_json.update(opts_dict)
    opts_dict_to_json.update(args_e4s_json)
    opts_dict_to_json["noise"] = str(opts_dict_to_json["noise"])
    opts_dict_to_json['CL_layers'] = WM_layer_list
    if opts.local_rank == 0:
        with open(os.path.join(output_dir, "train_opts.json"), "w") as f:
            json.dump(opts_dict_to_json, f, indent=4, sort_keys=True)
    # logging.info(pprint.pformat(vars(hidden_config)))

    opts.output_dir = output_dir
    # noise_config = [
    #     "Identity()",
    #     "JpegTest()",
    #     "Resize()",
    #     "GaussianBlur()",
    #     "MedianBlur()",
    #     "Brightness()",
    #     "Contrast()",
    #     "Saturation()",
    #     "Hue()",
    #     "SaltPepper()",
    #     "GaussianNoise()",
    # ]
    noise_config = [
        "Identity()",
    ]
    train = Train(opts, noise_config, None, args_e4s, device)
    # train.debug()
    train.debuging()


if __name__ == "__main__":
    main()
