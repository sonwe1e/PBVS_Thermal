import torch
import lightning.pytorch as pl
import heavyball
import torchmetrics.functional.image as tmi
import torch.nn.functional as F
from tools.losses import *

torch.set_float32_matmul_precision("high")


class EMA:
    """指数移动平均实现，支持不同数据类型"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """注册模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 使用全精度浮点数存储EMA参数，避免精度问题
                self.shadow[name] = param.data.clone().float().detach()

    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # 确保数据类型匹配，计算时先转为float
                param_float = param.data.float().detach()
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param_float
                )
                self.shadow[name] = new_average.detach()

    def apply_shadow(self):
        """应用EMA参数到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                # 确保数据类型匹配
                param.data.copy_(self.shadow[name].to(param.data.dtype))

    def restore(self):
        """恢复原始模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader):
        super().__init__()
        self.learning_rate = opt.learning_rate  # 学习率
        self.len_trainloader = len_trainloader  # 训练数据加载器长度
        self.opt = opt  # 配置参数
        self.model = model  # 模型
        self.mse_loss = torch.nn.MSELoss()  # 均方误差损失函数
        self.l1_loss = torch.nn.L1Loss()  # L1损失函数
        self.fft_loss = FFTLoss()
        self.ssim_loss = SSIMLoss()

        # 初始化EMA
        self.use_ema = getattr(opt, "use_ema", True)
        self.ema_decay = getattr(opt, "ema_decay", 0.999)
        if self.use_ema:
            self.ema = None  # 延迟初始化EMA，等到模型参数部署到适当设备后
            self.ema_initialized = False

        # ckpt = torch.load(
        #     "/media/hdd/sonwe1e/Competition/ImageSuperResolution/checkpoints/epoch_497-loss_26.740.ckpt",
        #     weights_only=False,
        #     map_location="cpu",
        # )["state_dict"]
        # for k in list(ckpt.keys()):
        #     if not k.startswith("model."):
        #         new_k = "model." + k
        #         ckpt[new_k] = ckpt[k]
        #         del ckpt[k]
        #         continue
        #     if k not in self.state_dict():
        #         del ckpt[k]
        # self.model.load_state_dict(ckpt, strict=False)
        # del ckpt

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
            total_steps=self.len_trainloader * self.opt.epochs // len(self.opt.devices),
            pct_start=self.opt.pct_start,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }

    def _init_ema(self):
        """初始化EMA"""
        if self.use_ema and not self.ema_initialized:
            self.ema = EMA(self.model, decay=self.ema_decay)
            self.ema_initialized = True

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 确保EMA已初始化
        if self.use_ema and not self.ema_initialized:
            self._init_ema()

        lr_image, hr_image = (batch["lr_image"], batch["hr_image"])
        # if torch.rand(1) < 0.33:
        #     lam = torch.rand(1, device=lr_image.device)
        #     index = torch.randperm(lr_image.size(0), device=lr_image.device)
        #     lr_image = lam * lr_image + (1 - lam) * lr_image[index, :]
        #     hr_image = lam * hr_image + (1 - lam) * hr_image[index, :]
        # Data augmentation with rotation and flipping
        if self.training and torch.rand(1) < 0.5:
            # Random rotation (0, 90, 180, or 270 degrees)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                lr_image = torch.rot90(lr_image, k, dims=[2, 3])
                hr_image = torch.rot90(hr_image, k, dims=[2, 3])

            # Random horizontal flip
            if torch.rand(1) < 0.5:
                lr_image = torch.flip(lr_image, dims=[3])
                hr_image = torch.flip(hr_image, dims=[3])

            # Random vertical flip
            if torch.rand(1) < 0.5:
                lr_image = torch.flip(lr_image, dims=[2])
                hr_image = torch.flip(hr_image, dims=[2])

        prediction = self(lr_image)  # 前向传播
        l1_loss = self.l1_loss(prediction, hr_image)  # L1损失
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

        # 更新EMA参数
        if self.use_ema and self.ema_initialized:
            self.ema.update()

        return loss

    def on_validation_start(self):
        """验证开始时应用EMA参数"""
        if self.use_ema and self.ema_initialized:
            self.ema.apply_shadow()

    def on_validation_end(self):
        """验证结束时恢复原始参数"""
        if self.use_ema and self.ema_initialized:
            self.ema.restore()

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        lr_image, hr_image = (batch["lr_image"], batch["hr_image"])
        prediction = self(lr_image)  # 前向传播
        l1_loss = self.l1_loss(prediction, hr_image)  # L1损失
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
        mse_loss = self.mse_loss(prediction, hr_image)
        rmse = torch.sqrt(mse_loss)
        psnr = tmi.peak_signal_noise_ratio(prediction, hr_image, data_range=(0, 1))
        ssim = tmi.structural_similarity_index_measure(
            prediction, hr_image, data_range=(0, 1)
        )

        self.log(f"metric/{mode}_psnr", psnr, on_step=False, on_epoch=True)
        self.log(f"metric/{mode}_ssim", ssim, on_step=False, on_epoch=True)
        self.log(f"metric/{mode}_rmse", rmse, on_step=False, on_epoch=True)

    def on_save_checkpoint(self, checkpoint):
        """保存checkpoint时保存EMA参数"""
        if self.use_ema and self.ema_initialized:
            # 统一使用float32存储EMA状态
            checkpoint["ema_state_dict"] = {
                k: v.clone().float().cpu() for k, v in self.ema.shadow.items()
            }

    def on_load_checkpoint(self, checkpoint):
        """加载checkpoint时加载EMA参数"""
        if self.use_ema and "ema_state_dict" in checkpoint:
            self._init_ema()
            # 加载已保存的EMA状态
            for k, v in checkpoint["ema_state_dict"].items():
                if k in self.ema.shadow:
                    self.ema.shadow[k] = v.to(self.device)
