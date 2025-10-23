from argparse import ArgumentParser
from .noise_argparser import NoiseArgParser


class TrainingOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument("--seed", default=63, type=int)

        self.parser.add_argument(
            "--noise",
            nargs="*",
            action=NoiseArgParser,
            help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'",
        )
        self.parser.add_argument("--des", default="Default", type=str)
        self.parser.add_argument("--max_epoch", default=80, type=int)
        self.parser.add_argument("--display_num", default=2, type=int)
        self.parser.add_argument("--batch_size", default=16, type=int)
        self.parser.add_argument("--test_batch_size", default=4, type=int)
        self.parser.add_argument("--num_workers", default=10, type=int)
        self.parser.add_argument("--max_train_iters", default=99999, type=int)
        self.parser.add_argument("--max_val_iters", default=999, type=int)
        self.parser.add_argument("--save_interval_epoch", default=50, type=int)
        self.parser.add_argument(
            "--optim_name", default="adam", type=str, help="Which optimizer to use"
        )
        self.parser.add_argument("--info", default=None, type=str)
        self.parser.add_argument("--seq_weight", default=0.1, type=float)
        self.parser.add_argument("--idvec_weight", default=1.0, type=float)

        # self.parser.add_argument("--seq_style_weight", default=0.1, type=float)
        # self.parser.add_argument("--stylevec_weight", default=1.0, type=float)

        self.parser.add_argument("--facenet_mode", default="arcface", type=str)
        self.parser.add_argument("--facenet_dir", default="./saved_models", type=str)

        # self.parser.add_argument('--aadblocks_dir', default='./saved_models/AAD_best.pth', type=str)
        # self.parser.add_argument('--attencoder_dir', default='./saved_models/Att_best.pth', type=str)
        # self.parser.add_argument('--discriminator_dir', default='./saved_models/Dis_best.pth', type=str)
        # self.parser.add_argument("--aadblocks_dir", default="./saved_models", type=str)
        # self.parser.add_argument("--attencoder_dir", default="./saved_models", type=str)
        # self.parser.add_argument(
        #     "--discriminator_dir", default="./saved_models", type=str
        # )

        self.parser.add_argument("--lr", default=1e-4, type=float)
        # self.parser.add_argument("--start_lr", default=1e-6, type=float)
        # self.parser.add_argument("--warm_step", default=3, type=int)
        # self.parser.add_argument("--min_lr", default=1e-6, type=float)

        # self.parser.add_argument("--adv_weight", default=0.1, type=float)
        # self.parser.add_argument("--att_weight", default=10.0, type=float)
        self.parser.add_argument("--id_weight", default=1.0, type=float)
        self.parser.add_argument("--rec_weight", default=10.0, type=float)
        # self.parser.add_argument("--msg_weight", default=1, type=float)
        # self.parser.add_argument("--robust_msg_weight", default=1, type=float)
        self.parser.add_argument("--wm_weight", default=10.0, type=float)

        self.parser.add_argument("--idloss_mode", default="Cos", type=str)
        self.parser.add_argument("--recloss_mode", default="lpips", type=str)
        self.parser.add_argument("--msgloss_mode", default="MSE", type=str)
        # self.parser.add_argument('--wmloss_mode', default='l2', type=str)

        self.parser.add_argument("--board_interval", default=100, type=int)
        self.parser.add_argument("--save_interval", default=5000, type=int)
        self.parser.add_argument("--image_interval", default=500, type=int)

        self.parser.add_argument("--exp_dir", default="./experiment", type=str)
        # self.parser.add_argument(
        #     "--trainimg_dir", default="/home/jpq/data/CelebA-HQ-img/", type=str
        # )  # --trainimg_dir /home/jpq/data/FFHQ-70K
        # celeba_dataset_root ffhq_dataset_root dataset_name
        self.parser.add_argument("--dataset_name", default="cahq", type=str)
        self.parser.add_argument(
            "--celeba_dataset_root", default="/home/gdata/face/CelebAMask-HQ", type=str
        )
        self.parser.add_argument("--ffhq_dataset_root", default="/home/gdata/face/FFHQ", type=str)
        # self.parser.add_argument("--valimg_dir", default="./validate_image", type=str)

        # self.parser.add_argument(
        #     "--size",
        #     "-s",
        #     default=256,
        #     type=int,
        #     help="The size of the images (images are square so this is height and width).",
        # )
        self.parser.add_argument(
            "--message",
            "-m",
            default=9,
            type=int,
            help="The length in bits of the watermark.",
        )
        # self.parser.add_argument(
        #     "--message_robust",
        #     "-mr",
        #     default=128,
        #     type=int,
        #     help="The length in bits of the watermark.",
        # )

        # self.parser.add_argument(
        #     "--attention", default="se", type=str, help="se,cbam,coord"
        # )

        # self.parser.add_argument(
        #     "--enable-fp16",
        #     dest="enable_fp16",
        #     action="store_true",
        #     help="Enable mixed-precision training.",
        # )
        self.parser.add_argument("--local_rank", default=-1)

    def parse(self):
        opts = self.parser.parse_args()
        return opts


class InjectionAndEvaluationOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument("--seed", default=0, type=int)
        self.parser.add_argument("--rand_select", default="Yes", type=str)

        self.parser.add_argument("--max_num", default=1000, type=int)
        self.parser.add_argument("--batch_size", default=10, type=int)
        self.parser.add_argument("--num_workers", default=2, type=int)

        self.parser.add_argument("--seq_weight", default=0.1, type=float)
        self.parser.add_argument("--idvec_weight", default=1.0, type=float)
        self.parser.add_argument("--seq_style_weight", default=0.1, type=float)
        self.parser.add_argument("--stylevec_weight", default=1.0, type=float)
        self.parser.add_argument("--recloss_mode", default="lpips", type=str)
        self.parser.add_argument("--rec_weight", default=1.0, type=float)
        self.parser.add_argument("--facenet_mode", default="arcface", type=str)
        self.parser.add_argument("--facenet_dir", default="./saved_models", type=str)

        exp_dir = "20240507_H0908_sw-0.15_mw-2.0_mm-MSE"  # 20231207_H1131_sw-0.1_mw-1.5_mm-MSE_True 20231220_H1141_sw-0.1_mw-1.5_mm-MSE_True
        epoch_n = "00060"
        # self.parser.add_argument(
        #     "--aadblocks_dir",
        #     default=f"./experiment/{exp_dir}/CheckPoints/AAD_{epoch_n}.pth",
        #     type=str,
        # )
        # self.parser.add_argument(
        #     "--attencoder_dir",
        #     default=f"./experiment/{exp_dir}/CheckPoints/Att_{epoch_n}.pth",
        #     type=str,
        # )
        # self.parser.add_argument(
        #     "--encoder_dir",
        #     default=f"./experiment/{exp_dir}/CheckPoints/Enc_{epoch_n}.pth",
        #     type=str,
        # )
        # self.parser.add_argument(
        #     "--decoder_dir",
        #     default=f"./experiment/{exp_dir}/CheckPoints/Dec_{epoch_n}.pth",
        #     type=str,
        # )

        self.parser.add_argument(
            "--aadblocks_dir",
            default=f"./experiment/{exp_dir}/BestResult/AAD_best.pth",
            type=str,
        )
        self.parser.add_argument(
            "--attencoder_dir",
            default=f"./experiment/{exp_dir}/BestResult/Att_best.pth",
            type=str,
        )
        self.parser.add_argument(
            "--encoder_dir",
            default=f"./experiment/{exp_dir}/BestResult/Enc_best.pth",
            type=str,
        )
        self.parser.add_argument(
            "--decoder_dir",
            default=f"./experiment/{exp_dir}/BestResult/Dec_best.pth",
            type=str,
        )

        self.parser.add_argument("--seq_type", default="mls", type=str)

        self.parser.add_argument("--exp_dir", default="./experiment", type=str)
        self.parser.add_argument(
            "--img_dir", default="/home/jpq/data/CelebA-HQ-img", type=str
        )  # /home/jpq/data/CelebA-HQ-img  /home/jpq/data/FFHQ-70K

        self.parser.add_argument(
            "--size",
            "-s",
            default=256,
            type=int,
            help="The size of the images (images are square so this is height and width).",
        )
        self.parser.add_argument(
            "--message",
            "-m",
            default=9,
            type=int,
            help="The length in bits of the watermark.",
        )
        self.parser.add_argument(
            "--message_robust",
            "-mr",
            default=64,
            type=int,
            help="The length in bits of the watermark.",
        )
        self.parser.add_argument(
            "--enable-fp16",
            dest="enable_fp16",
            action="store_true",
            help="Enable mixed-precision training.",
        )
        self.parser.add_argument("--peak_threshold", type=int, default=5)
        self.parser.add_argument(
            "--attention", default="coord", type=str, help="se,cbam,coord"
        )
        self.parser.add_argument(
            "--psp_checkpoint_path",
            default="saved_models/psp_ffhq_encode.pt",
            type=str,
            help="Path to pSp model checkpoint",
        )

    def parse(self):
        opts = self.parser.parse_args()
        return opts


class InjectionOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument("--seed", default=0, type=int)
        self.parser.add_argument("--rand_select", default="Yes", type=str)

        self.parser.add_argument("--max_num", default=500, type=int)
        self.parser.add_argument("--batch_size", default=20, type=int)
        self.parser.add_argument("--num_workers", default=8, type=int)

        self.parser.add_argument("--seq_weight", default=0.1, type=float)
        self.parser.add_argument("--idvec_weight", default=1.0, type=float)

        self.parser.add_argument("--facenet_mode", default="arcface", type=str)
        self.parser.add_argument("--facenet_dir", default="./saved_models", type=str)

        self.parser.add_argument(
            "--aadblocks_dir",
            default="./experiment/CelebAHQ_MSE_Noiser_DF/BestResult/AAD_best.pth",
            type=str,
        )
        self.parser.add_argument(
            "--attencoder_dir",
            default="./experiment/CelebAHQ_MSE_Noiser_DF/BestResult/Att_best.pth",
            type=str,
        )
        self.parser.add_argument(
            "--encoder_dir",
            default="./experiment/CelebAHQ_MSE_Noiser_DF/BestResult/Enc_best.pth",
            type=str,
        )
        self.parser.add_argument(
            "--decoder_dir",
            default="./experiment/CelebAHQ_MSE_Noiser_DF/BestResult/Dec_best.pth",
            type=str,
        )
        self.parser.add_argument("--seq_type", default="mls", type=str)

        self.parser.add_argument("--exp_dir", default="./experiment", type=str)
        self.parser.add_argument(
            "--img_dir", default="/home/jpq/data/CelebA-HQ-img", type=str
        )  # /home/jpq/data/CelebA-HQ-img  /home/jpq/data/FFHQ-70K

        self.parser.add_argument(
            "--size",
            "-s",
            default=256,
            type=int,
            help="The size of the images (images are square so this is height and width).",
        )
        self.parser.add_argument(
            "--message",
            "-m",
            default=9,
            type=int,
            help="The length in bits of the watermark.",
        )
        self.parser.add_argument(
            "--enable-fp16",
            dest="enable_fp16",
            action="store_true",
            help="Enable mixed-precision training.",
        )

    def parse(self):
        opts = self.parser.parse_args()
        return opts


class EvaluationOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument("--batch_size", type=int, default=20)
        self.parser.add_argument("--num_workers", default=6, type=int)
        self.parser.add_argument("--peak_threshold", type=int, default=5)
        self.parser.add_argument(
            "--seq_dir", type=str, default="./experiment/FFHQ-sw(0.1)_st(mls)_arcface"
        )
        self.parser.add_argument("--facenet_mode", type=str, default="arcface")
        self.parser.add_argument("--facenet_dir", type=str, default="./saved_models")
        self.parser.add_argument(
            "--imgpos_dir",
            type=str,
            default="./experiment/FFHQ-sw(0.1)_st(mls)_arcface/reconstructed",
        )
        self.parser.add_argument(
            "--imgneg_dir",
            type=str,
            default="./experiment/FFHQ-sw(0.1)_st(mls)_arcface/reconstructed_DF",
        )
        self.parser.add_argument(
            "--decoder_dir",
            default="./experiment/CelebAHQ_MSE_Noiser_DF/BestResult/Dec_best.pth",
            type=str,
        )
        self.parser.add_argument(
            "--size",
            "-s",
            default=256,
            type=int,
            help="The size of the images (images are square so this is height and width).",
        )
        self.parser.add_argument(
            "--message",
            "-m",
            default=9,
            type=int,
            help="The length in bits of the watermark.",
        )
        self.parser.add_argument(
            "--enable-fp16",
            dest="enable_fp16",
            action="store_true",
            help="Enable mixed-precision training.",
        )

    def parse(self):
        opts = self.parser.parse_args()
        return opts
