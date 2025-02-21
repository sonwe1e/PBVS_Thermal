import torch
import lightning.pytorch as pl
import heavyball
import torchmetrics.functional.image as tmi
import torch.nn.functional as F
from tools.losses import *

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate  # 学习率
        self.len_trainloader = len_trainloader  # 训练数据加载器长度
        self.opt = opt  # 配置参数
        self.model = model  # 模型
        self.mse_loss = torch.nn.MSELoss()  # 均方误差损失函数
        self.l1_loss = torch.nn.L1Loss()  # 交叉熵损失函数
        self.fft_loss = FFTLoss()
        self.ssim_loss = SSIMLoss()
        # self.semantic_loss = LPIPS()
        ckpt = torch.load(
            "/media/hdd/sonwe1e/Competition/PBVS_Thermal/checkpoints/epoch_405-loss_26.706.ckpt",
            weights_only=False,
            map_location="cpu",
        )["state_dict"]
        for k in list(ckpt.keys()):
            if "loss" in k:
                ckpt.pop(k)
            if "model." in k:
                ckpt[k.replace("model.", "")] = ckpt.pop(k)
        ckpt.pop("to_img.0.weight")
        self.model.load_state_dict(ckpt, strict=False)

    def forward(self, x):
        """前向传播"""
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        """配置优化器和学习率 Scheduler"""
        # self.optimizer = heavyball.ForeachAdamW(
        #     self.parameters(),
        #     weight_decay=self.opt.weight_decay,
        #     lr=self.learning_rate,
        #     betas=(0.9, self.opt.beta2),
        #     caution=True,
        # )
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
            betas=(0.9, self.opt.beta2),
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.len_trainloader * self.opt.epochs,
            pct_start=self.opt.pct_start,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        lr_image, hr_image = (batch["lr_image"], batch["hr_image"])
        if torch.rand(1) < 0.5:
            lam = torch.rand(1, device=lr_image.device)
            index = torch.randperm(lr_image.size(0), device=lr_image.device)
            lr_image = lam * lr_image + (1 - lam) * lr_image[index, :]
            hr_image = lam * hr_image + (1 - lam) * hr_image[index, :]

        prediction = self(lr_image)  # 前向传播
        l1_loss = self.l1_loss(prediction, hr_image)  # 交叉熵损失
        fft_loss = self.fft_loss(prediction, hr_image)
        ssim_loss = self.ssim_loss(prediction, hr_image)
        loss = l1_loss + fft_loss + ssim_loss

        # 记录训练损失及指标
        self.log("loss/train_loss", loss)
        self.log("loss/train_l1_loss", l1_loss)
        self.log("loss/train_fft_loss", fft_loss)
        self.log("loss/train_ssim_loss", ssim_loss)
        self.log("trainer/learning_rate", self.optimizer.param_groups[0]["lr"])
        self.calculate_metrics(prediction, hr_image, "train")
        return loss  # + 0.003 * self.semantic_loss(prediction, hr_image)

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        lr_image, hr_image = (batch["lr_image"], batch["hr_image"])
        prediction = self(lr_image)  # 前向传播
        l1_loss = self.l1_loss(prediction, hr_image)  # 交叉熵损失
        fft_loss = self.fft_loss(prediction, hr_image)
        ssim_loss = self.ssim_loss(prediction, hr_image)
        loss = l1_loss + fft_loss + ssim_loss

        # 记录验证损失及指标
        self.log("loss/valid_loss", loss)
        self.log("loss/valid_l1_loss", l1_loss)
        self.log("loss/valid_fft_loss", fft_loss)
        self.log("loss/valid_ssim_loss", ssim_loss)
        self.calculate_metrics(prediction, hr_image, "valid")

    def calculate_metrics(self, prediction, hr_image, mode):
        hr_max, hr_min = torch.max(hr_image), torch.min(hr_image)
        if hr_max - hr_min == 1:
            prediction = torch.clamp(prediction, 0, 1)
        elif hr_max - hr_min == 2:
            prediction = torch.clamp(prediction, -1, 1)
            prediction = prediction * 0.5 + 0.5
            hr_image = hr_image * 0.5 + 0.5
        else:
            prediction = torch.clamp(prediction, hr_min, hr_max)
            hr_image = (hr_image - hr_min) / (hr_max - hr_min)
            prediction = (prediction - hr_min) / (hr_max - hr_min)
        mse_loss = self.mse_loss(prediction, hr_image)
        rmse = torch.sqrt(mse_loss)
        psnr = tmi.peak_signal_noise_ratio(prediction, hr_image, data_range=1)
        ssim = tmi.structural_similarity_index_measure(
            prediction, hr_image, data_range=1
        )
        if mode == "valid":
            prediction, hr_image = (prediction * 2) - 1, (hr_image * 2) - 1
            alex = tmi.learned_perceptual_image_patch_similarity(
                prediction, hr_image, "alex"
            )
            vgg = tmi.learned_perceptual_image_patch_similarity(
                prediction, hr_image, "vgg"
            )
            squeeze = tmi.learned_perceptual_image_patch_similarity(
                prediction, hr_image, "squeeze"
            )
            self.log(f"metric/{mode}_alex", alex, on_step=False, on_epoch=True)
            self.log(f"metric/{mode}_vgg", vgg, on_step=False, on_epoch=True)
            self.log(f"metric/{mode}_squeeze", squeeze, on_step=False, on_epoch=True)

        self.log(f"metric/{mode}_psnr", psnr, on_step=False, on_epoch=True)
        self.log(f"metric/{mode}_ssim", ssim, on_step=False, on_epoch=True)
        self.log(f"metric/{mode}_rmse", rmse, on_step=False, on_epoch=True)
