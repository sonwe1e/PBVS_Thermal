import torch
import heavyball
import lightning.pytorch as pl
import torchmetrics.functional.image as tmi
from tools.losses import *


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

        if hasattr(opt, "checkpoint_path"):
            self.load_checkpoint(opt.checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """
        从checkpoint文件加载预训练权重到模型

        Args:
            checkpoint_path (str): checkpoint文件路径
        """
        if not checkpoint_path or not isinstance(checkpoint_path, str):
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            # 加载权重
            ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")[
                "state_dict"
            ]

            # 处理前缀
            new_ckpt = {}
            for k, v in ckpt.items():
                if "model." in k:
                    new_k = k.replace("model.", "")
                    new_ckpt[new_k] = v

            # 加载到模型
            if new_ckpt:
                self.model.load_state_dict(new_ckpt, strict=True)
                print(f"Successfully loaded checkpoint from {checkpoint_path}")
            else:
                print(f"No valid weights found in checkpoint {checkpoint_path}")

            # 释放内存
            del ckpt, new_ckpt
        except Exception as e:
            print(f"Failed to load checkpoint: {str(e)}")

    def forward(self, x):
        """前向传播"""
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        """配置优化器和学习率 Scheduler"""
        self.optimizer = heavyball.ForeachAdamW(
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
        if self.use_ema and not self.ema_initialized:
            self._init_ema()
        if isinstance(batch, dict):
            lr_image, hr_image = (batch["lr_image"], batch["hr_image"])
        else:
            lr_image, hr_image = batch[0], batch[1]

        prediction = self(lr_image)  # 前向传播
        l1_loss = self.l1_loss(prediction, hr_image)  # L1损失
        fft_loss = self.fft_loss(prediction, hr_image)
        ssim_loss = self.ssim_loss(prediction, hr_image)
        loss = l1_loss + fft_loss + ssim_loss

        # 记录训练损失及指标
        self.log("loss/train_loss", loss, sync_dist=True)
        self.log("loss/train_l1_loss", l1_loss, sync_dist=True)
        self.log("loss/train_fft_loss", fft_loss, sync_dist=True)
        self.log("loss/train_ssim_loss", ssim_loss, sync_dist=True)
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
        if isinstance(batch, dict):
            lr_image, hr_image = (batch["lr_image"], batch["hr_image"])
        else:
            lr_image, hr_image = batch[0], batch[1]
        prediction = self(lr_image)  # 前向传播
        l1_loss = self.l1_loss(prediction, hr_image)  # L1损失
        fft_loss = self.fft_loss(prediction, hr_image)
        ssim_loss = self.ssim_loss(prediction, hr_image)
        loss = l1_loss + fft_loss + ssim_loss

        # 记录验证损失及指标
        self.log("loss/valid_loss", loss, sync_dist=True)
        self.log("loss/valid_l1_loss", l1_loss, sync_dist=True)
        self.log("loss/valid_fft_loss", fft_loss, sync_dist=True)
        self.log("loss/valid_ssim_loss", ssim_loss, sync_dist=True)
        self.calculate_metrics(prediction, hr_image, "valid")

    def rgb_to_y(self, image):
        """将 RGB 图像转换为 Y 通道 (tensor)，支持批量操作。

        Args:
            image: (Tensor) [B, C, H, W] 或 [C, H, W] 范围在 [0, 1] 的 RGB 图像。

        Returns:
            (Tensor) [B, 1, H, W] 或 [1, H, W] Y 通道图像。
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)  # 添加批次维度
        assert image.shape[1] == 3, "输入图像必须是 RGB 格式"

        # 调整权重系数的形状，使其可以与图像张量进行广播相乘
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device).view(
            1, 3, 1, 1
        )
        y = (image * weights).sum(dim=1, keepdim=True)
        return y

    def calculate_metrics(self, prediction, hr_image, mode):
        """计算 Y 通道上的 PSNR 和 SSIM。"""
        if prediction.max() > 1.0:
            prediction = torch.clamp(prediction, 0.0, 1.0)

        # 转换为 Y 通道
        prediction_y = self.rgb_to_y(prediction)
        hr_image_y = self.rgb_to_y(hr_image)

        # 计算 PSNR 和 SSIM
        psnr = tmi.peak_signal_noise_ratio(prediction_y, hr_image_y, data_range=(0, 1))
        ssim = tmi.structural_similarity_index_measure(
            prediction_y, hr_image_y, data_range=(0, 1)
        )

        # 记录指标
        self.log(
            f"metric/{mode}_psnr", psnr, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            f"metric/{mode}_ssim", ssim, on_step=False, on_epoch=True, sync_dist=True
        )

    def on_save_checkpoint(self, checkpoint):
        """保存checkpoint时保存EMA参数"""
        if self.use_ema and self.ema_initialized:
            checkpoint["ema_state_dict"] = {
                k: v.clone().float().cpu() for k, v in self.ema.shadow.items()
            }

    def on_load_checkpoint(self, checkpoint):
        """加载checkpoint时加载EMA参数"""
        if self.use_ema and "ema_state_dict" in checkpoint:
            self._init_ema()
            for k, v in checkpoint["ema_state_dict"].items():
                if k in self.ema.shadow:
                    self.ema.shadow[k] = v.to(self.device)
