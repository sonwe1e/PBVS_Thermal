import torch
import torch.nn as nn
import lpips
import torch.nn.functional as F


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=0.1, reduction="mean"):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)


class FFTLossv1(nn.Module):
    def __init__(self, loss_weight=0.1, reduction="mean"):
        super(FFTLossv1, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        # 分离幅值和相位
        pred_amp, pred_phase = torch.abs(pred_fft), torch.angle(pred_fft)
        target_amp, target_phase = torch.abs(target_fft), torch.angle(target_fft)

        # 幅值损失 + 相位余弦相似度损失
        amp_loss = self.criterion(pred_amp, target_amp)
        phase_loss = 1 - torch.cos(pred_phase - target_phase).mean()

        return self.loss_weight * (amp_loss + 0.5 * phase_loss)  # 可调权重


class GradientLoss(nn.Module):
    def __init__(self, scales=3, alpha=0.5):
        super(GradientLoss, self).__init__()
        self.scales = scales
        self.alpha = alpha

    def forward(self, pred, target):
        loss = 0.0
        for scale in range(self.scales):
            # 下采样到不同尺度
            if scale > 0:
                pred_down = F.avg_pool2d(pred, kernel_size=2**scale)
                target_down = F.avg_pool2d(target, kernel_size=2**scale)
            else:
                pred_down, target_down = pred, target

            # 计算水平和垂直梯度
            pred_grad_x = pred_down[:, :, :, 1:] - pred_down[:, :, :, :-1]
            pred_grad_y = pred_down[:, :, 1:, :] - pred_down[:, :, :-1, :]
            target_grad_x = target_down[:, :, :, 1:] - target_down[:, :, :, :-1]
            target_grad_y = target_down[:, :, 1:, :] - target_down[:, :, :-1, :]

            # 梯度差损失（L1范数）
            loss += self.alpha**scale * (
                F.l1_loss(pred_grad_x, target_grad_x)
                + F.l1_loss(pred_grad_y, target_grad_y)
            )
        return loss / self.scales


class LPIPS(nn.Module):
    def __init__(self, net="vgg", pretrained=True):  # 默认使用 VGG
        super(LPIPS, self).__init__()
        self.loss_fn = lpips.LPIPS(net=net, pretrained=pretrained, verbose=False)

    def forward(self, x, y):
        """
        计算 LPIPS 损失。

        Args:
            x: 输入图像，形状为 [B, C, H, W]，范围应为 [-1, 1] 或 [0, 1]（lpips 会自动处理）
            y: 目标图像，形状为 [B, C, H, W]，范围应为 [-1, 1] 或 [0, 1]
        Returns:
            LPIPS 损失，一个标量。
        """
        # 不需要手动归一化，lpips内部做了
        return self.loss_fn(x, y).mean()  # lpips 返回的是 [B, 1, 1, 1]，取平均得到标量


class SemanticLoss(nn.Module):
    def __init__(self, lpips_net="dinov2_vits14"):
        super().__init__()
        self.semantic_model = torch.hub.load("facebookresearch/dinov2", lpips_net)

    @torch.no_grad()
    def forward(self, x, y):
        f_x = self.semantic_model(x)
        f_y = self.semantic_model(y)
        return torch.nn.functional.l1_loss(f_x, f_y)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, k1=0.01, k2=0.03, data_range=1):
        """
        SSIM Loss构造函数。

        参数:
            window_size (int): 高斯窗口大小，默认为11。
            sigma (float): 高斯核的标准差，默认为1.5。
            k1, k2 (float): SSIM计算中的稳定性常数，默认为0.01和0.03。
            data_range (float): 输入数据的范围，例如[0,1]时为1，[0,255]时为255。
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range

        # 创建高斯核并注册为不可训练的缓冲区
        self.register_buffer("gaussian_kernel", self._create_gaussian_kernel())

    def _create_gaussian_kernel(self):
        """生成二维高斯核。"""
        coords = torch.arange(self.window_size, dtype=torch.float)
        coords -= self.window_size // 2  # 居中坐标

        # 计算一维高斯分布
        g = torch.exp(-(coords**2) / (2 * self.sigma**2))
        g /= g.sum()  # 归一化

        # 生成二维高斯核
        g_2d = g.unsqueeze(1) * g.unsqueeze(0)  # (window_size, window_size)
        g_2d = g_2d.unsqueeze(0).unsqueeze(
            0
        )  # 扩展维度 (1, 1, window_size, window_size)
        return g_2d

    def forward(self, x, y):
        """
        计算输入图像x和目标图像y之间的SSIM损失。

        参数:
            x (torch.Tensor): 预测图像，形状为(B, C, H, W)。
            y (torch.Tensor): 真实图像，形状与x相同。

        返回:
            torch.Tensor: SSIM损失值。
        """
        if x.shape != y.shape:
            raise ValueError("输入x和y的形状必须相同")

        batch_size, channels, height, width = x.shape

        # 扩展高斯核以匹配输入通道数
        kernel = self.gaussian_kernel.repeat(
            channels, 1, 1, 1
        )  # (C, 1, window_size, window_size)
        padding = self.window_size // 2

        # 计算局部均值
        mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
        mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)

        # 计算均值平方和交叉项
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        # 计算方差和协方差
        sigma_x_sq = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy

        # 计算稳定性常数
        C1 = (self.k1 * self.data_range) ** 2
        C2 = (self.k2 * self.data_range) ** 2

        # 计算SSIM的分子和分母
        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

        # 计算SSIM映射并取平均
        ssim_map = numerator / (denominator + 1e-8)  # 避免除以零
        ssim = ssim_map.mean()

        # 损失为1 - SSIM
        return 1 - ssim


class GradientHistLoss(nn.Module):
    def __init__(self, bins=64, margin=0.4):
        super().__init__()
        self.bins = bins
        self.margin = margin  # 重要梯度区间扩展系数

    def dynamic_bin_range(self, magnitudes):
        # 动态计算重要梯度区间
        max_val = magnitudes.quantile(0.95)
        return torch.linspace(0, max_val.item(), self.bins + 1)

    def compute_hist(self, mag, bins):
        # 可微分直方图计算
        weights = 1 - torch.abs(mag.unsqueeze(-1) - bins[:-1]) / (bins[1] - bins[0])
        weights = torch.clamp(weights, 0, 1)
        hist = torch.sum(weights, dim=[0, 1, 2])
        return hist / hist.sum()

    def forward(self, pred_grad, gt_grad):
        loss = 0
        for p_mag, g_mag in zip(pred_grad, gt_grad):
            # 动态范围计算
            dyn_bins = self.dynamic_bin_range(g_mag)
            dyn_bins = dyn_bins.to(p_mag.device)
            # 直方图计算
            p_hist = self.compute_hist(p_mag, dyn_bins)
            g_hist = self.compute_hist(g_mag, dyn_bins)

            # 重要区域加权
            weight = torch.exp(self.margin * torch.arange(self.bins) / self.bins).to(
                p_mag.device
            )
            loss += F.l1_loss(p_hist * weight, g_hist * weight)
        return loss / len(pred_grad)


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        """
        x: 预测值
        y: 真实值
        """
        loss = torch.sqrt((x - y) ** 2 + self.epsilon**2)
        return torch.mean(loss)


# 示例用法 (假设你的图像已经在 [0, 1] 范围内):
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(4, 3, 256, 256).to(device)  # 示例输入
    y = torch.rand(4, 3, 256, 256).to(device)

    # 使用 VGG 作为 backbone
    lpips_loss_fn = LPIPS(net="vgg", use_gpu=torch.cuda.is_available()).to(device)
    lpips_loss = lpips_loss_fn(x, y)
    print("LPIPS Loss (VGG):", lpips_loss.item())

    # 使用 AlexNet 作为 backbone
    lpips_loss_fn_alex = LPIPS(net="alex", use_gpu=torch.cuda.is_available()).to(
        device
    )  # 实例化一个新的 LPIPS
    lpips_loss_alex = lpips_loss_fn_alex(x, y)
    print("LPIPS Loss (AlexNet):", lpips_loss_alex.item())

    # 结合 L1 损失的示例
    l1_loss_fn = nn.L1Loss()
    l1_loss = l1_loss_fn(x, y)
    alpha = 0.5  # L1 损失的权重
    total_loss = lpips_loss + alpha * l1_loss
    print("Total Loss (LPIPS + L1):", total_loss.item())
