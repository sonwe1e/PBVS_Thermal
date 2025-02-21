import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import LayerNorm2d


class PALayer(nn.Module):
    def __init__(self, channel, bias=False):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel // 8, channel, 1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


def default_conv(in_channels, out_channels, kernel_size, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
    )


class CP_Attention_block(nn.Module):
    def __init__(self, dim, conv=default_conv, kernel_size=5, bias=False):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=bias)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=bias)
        self.calayer = CALayer(dim, bias)
        self.palayer = PALayer(dim, bias)

    def forward(self, x):
        res = self.act1(self.conv1(x)) + x
        res = self.palayer(self.calayer(self.conv2(res))) + x
        return res


class DMlp(nn.Module):
    """
    Dynamic Multilayer Perceptron (DMlp) 模块。

    该模块包含两个卷积层和一个 GELU 激活函数，用于特征变换。
    还包括一个 Dropout 层，用于防止过拟合。

    Args:
        dim (int): 输入特征的通道数。
        growth_rate (float, optional): 隐藏层通道相对于 dim 的增长比例。默认为 4.0。
        dropout_date (float, optional): Dropout 的概率。默认为 0.0。

    Input:
        x (torch.Tensor): 形状为 [B, C, H, W] 的输入特征图，其中 B 是 batch size，C 是通道数，H 是高度，W 是宽度。

    Output:
        torch.Tensor: 形状为 [B, C, H, W] 的输出特征图。
    """

    def __init__(self, dim, growth_rate=4.0, dropout_date=0.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)  # 1x1 卷积，用于增加通道数
        self.act = nn.GELU()  # GELU 激活函数
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)  # 1x1 卷积，用于恢复通道数
        self.dp = nn.Dropout(dropout_date)  # Dropout 层

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入特征图。

        Returns:
            torch.Tensor: 输出特征图。
        """
        x = self.conv_0(x)
        x = self.act(x)
        x = self.dp(x)
        x = self.conv_1(x)
        return x


class SuperResMoEFeatureExtractor(nn.Module):
    """
    超分辨率混合专家特征提取器 (Super-Resolution Mixture of Experts Feature Extractor) 模块。

    该模块使用多个 DMlp 专家，并通过一个门控机制将它们的输出融合起来。
    门控机制允许网络根据输入自适应地选择专家的组合。

    Args:
        dim (int): 输入特征的通道数（也为 DMlp 的输入输出通道数）。
        num_experts (int, optional): 专家数量。默认为 4。
        growth_rate (float, optional): DMlp 中隐藏层通道相对于 dim 的增长比例。默认为 4.0。
        dropout_date (float, optional): Dropout 的概率。默认为 0.0。

    Input:
        x (torch.Tensor): 形状为 [B, C, H, W] 的输入特征图。

    Output:
        torch.Tensor: 融合后的特征图，形状为 [B, C, H, W]。
    """

    def __init__(self, dim, num_experts=4, growth_rate=4.0, dropout_date=0.0):
        super().__init__()
        self.num_experts = num_experts

        # 构建多个 DMlp 专家
        self.experts = nn.ModuleList(
            [DMlp(dim, growth_rate, dropout_date) for _ in range(num_experts)]
        )

        self.gate = nn.Conv2d(
            dim, num_experts, kernel_size=1, stride=1, padding=0
        )  # 1x1 卷积，用于生成门控权重
        self.softmax = nn.Softmax(dim=1)  # Softmax 函数，用于归一化门控权重

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入特征图，形状为 [B, C, H, W]。

        Returns:
            torch.Tensor: 融合后的特征图，形状为 [B, C, H, W]。
        """
        # 1. 计算门控权重，形状为 [B, num_experts, H, W]
        gate_weights = self.gate(x)
        gate_weights = self.softmax(gate_weights)  # 在专家维度上归一化

        # 2. 分别计算各个专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        # 将各个专家输出堆叠到一起，形状变为 [B, num_experts, C, H, W]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 3. 利用门控权重对专家输出进行加权融合
        # 扩展 gate_weights 至 [B, num_experts, 1, H, W]，以便与专家输出逐元素相乘
        gate_weights = gate_weights.unsqueeze(2)
        weighted_outputs = expert_outputs * gate_weights
        fused_feature = weighted_outputs.sum(dim=1)  # 汇聚到 [B, C, H, W]

        # 4. 可选：添加残差连接，有助于信息传递和训练稳定性
        output = fused_feature + x

        return output


class PCFN(nn.Module):
    """
    Partially Connected Feedforward Network (PCFN) 模块。

    该模块将输入特征分为两部分，只对其中一部分进行卷积操作，然后将两部分合并。
    这种部分连接的方式可以减少计算量，并提高模型的泛化能力。

    Args:
        dim (int): 输入特征的通道数。
        growth_rate (float, optional): 隐藏层通道相对于 dim 的增长比例。默认为 4.0。
        p_rate (float, optional): 用于分割通道的比例。默认为 0.25。
        dropout_rate (float, optional): Dropout 的概率。默认为 0.0。

    Input:
        x (torch.Tensor): 形状为 [B, C, H, W] 的输入特征图。

    Output:
        torch.Tensor: 形状为 [B, C, H, W] 的输出特征图。
    """

    def __init__(self, dim, growth_rate=4.0, p_rate=0.25, dropout_rate=0.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)  # 1x1 卷积，用于增加通道数
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)  # 3x3 卷积，只应用于部分通道

        self.act = nn.GELU()  # GELU 激活函数
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)  # 1x1 卷积，用于恢复通道数

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

        self.dp = nn.Dropout(dropout_rate)  # Dropout 层

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入特征图。

        Returns:
            torch.Tensor: 输出特征图。
        """
        if self.training:  # 训练模式
            x = self.act(self.conv_0(self.dp(x)))
            x1, x2 = torch.split(
                x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1
            )  # 分割通道
            x1 = self.act(self.conv_1(x1))  # 只对部分通道进行卷积
            x = self.conv_2(self.dp(torch.cat([x1, x2], dim=1)))  # 合并通道
        else:  # 推理模式
            x = self.act(self.conv_0(x))
            x[:, : self.p_dim, :, :] = self.act(
                self.conv_1(x[:, : self.p_dim, :, :])
            )  # 只对部分通道进行卷积
            x = self.conv_2(x)
        return x


class SMFA(nn.Module):
    """
    Spatial Modulation and Feature Aggregation (SMFA) 模块。

    该模块用于提取空间信息，并通过特征聚合增强特征表达。
    包含可学习参数 alpha 和 belt 用于空间调制。

    Args:
        dim (int, optional): 输入特征的通道数。默认为 36。
        nums_expert (int, optional): SuperResMoEFeatureExtractor 中专家的数量。默认为 4。
        growth_rate (int, optional): SuperResMoEFeatureExtractor 中增长率。默认为 4。
        dropout_rate (float, optional): Dropout 的概率。默认为 0.0。

    Input:
        f (torch.Tensor): 形状为 [B, C, H, W] 的输入特征图。

    Output:
        torch.Tensor: 形状为 [B, C, H, W] 的输出特征图。
    """

    def __init__(self, dim=36, nums_expert=4, growth_rate=4, dropout_rate=0.0):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)  # 1x1 卷积，用于增加通道数
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)  # 1x1 卷积
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)  # 1x1 卷积
        self.dp = nn.Dropout(dropout_rate)  # Dropout 层

        self.lde = SuperResMoEFeatureExtractor(
            dim, nums_expert, growth_rate, dropout_rate
        )  # 超分辨率混合专家特征提取器
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)  # 深度可分离卷积

        self.gelu = nn.GELU()  # GELU 激活函数
        self.down_scale = 8  # 降采样比例

        self.alpha = nn.Parameter(
            torch.ones((1, dim, 1, 1))
        )  # 可学习参数 alpha，用于空间调制
        self.belt = nn.Parameter(
            torch.zeros((1, dim, 1, 1))
        )  # 可学习参数 belt，用于空间调制

    def forward(self, f):
        """
        前向传播函数。

        Args:
            f (torch.Tensor): 输入特征图。

        Returns:
            torch.Tensor: 输出特征图。
        """
        _, _, h, w = f.shape
        f = self.dp(f)
        y, x = self.linear_0(f).chunk(2, dim=1)  # 分割通道
        x_s = self.dw_conv(
            F.adaptive_max_pool2d(
                x, (h // self.down_scale, w // self.down_scale)
            )  # 降采样
        )  # 深度可分离卷积
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)  # 计算方差
        x_l = x * F.interpolate(
            self.gelu(self.linear_1(self.dp(x_s * self.alpha + x_v * self.belt))),
            size=(h, w),
            mode="nearest",
        )  # 空间调制
        y_d = self.lde(y)  # 超分辨率混合专家特征提取
        return self.linear_2(self.dp(y_d + x_l))


class FMB(nn.Module):
    """
    Feature Modulation Block (FMB) 模块。

    该模块将 SMFA 和 PCFN 模块结合起来，用于更有效地进行特征调制和提取。

    Args:
        dim (int): 输入特征的通道数。
        smfa_expert (int, optional): SMFA 中专家的数量。默认为 4。
        smfa_growth (int, optional): SMFA 中增长率。默认为 4。
        pcfn_growth (int, optional): PCFN 中增长率。默认为 4。
        snfa_dropout (float, optional): SMFA 中的 Dropout 概率。默认为 0.0。
        pcfn_dropout (float, optional): PCFN 中的 Dropout 概率。默认为 0.0。
        p_rate (float, optional): PCFN 中用于分割通道的比例。默认为 0.25。

    Input:
        x (torch.Tensor): 形状为 [B, C, H, W] 的输入特征图。

    Output:
        torch.Tensor: 形状为 [B, C, H, W] 的输出特征图。
    """

    def __init__(
        self,
        dim,
        smfa_expert=4,
        smfa_growth=4,
        pcfn_growth=4,
        snfa_dropout=0.0,
        pcfn_dropout=0.0,
        p_rate=0.25,
    ):
        super().__init__()
        self.smfa = SMFA(dim, smfa_expert, smfa_growth, snfa_dropout)
        # self.cpa = CP_Attention_block(dim)  # CP 注意力块
        self.pcfn = PCFN(dim, pcfn_growth, p_rate, pcfn_dropout)  # 部分连接前馈网络
        self.ln1 = LayerNorm2d(dim)  # Layer Normalization
        self.ln2 = LayerNorm2d(dim)  # Layer Normalization

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入特征图。

        Returns:
            torch.Tensor: 输出特征图。
        """
        x = self.ln1(self.smfa(x)) + x
        # x = self.cpa(x)
        x = self.ln2(self.pcfn(x)) + x
        return x


class SMFANet(nn.Module):
    """
    Spatial Modulation and Feature Aggregation Network (SMFANet) 模型。

    该模型用于超分辨率任务，包含多个 FMB 模块，并使用 PixelShuffle 进行上采样。

    Args:
        dim (int, optional): 特征的通道数。默认为 96。
        n_blocks (int, optional): FMB 模块的数量。默认为 8。
        upscaling_factor (int, optional): 上采样因子。默认为 8。
        fmb_params (dict, optional):  包含 FMB 模块参数的字典。
            如果为 None，则使用 FMB 模块的默认参数。默认为 None.
            可以包含以下键：
                - 'smfa_expert' (int): SMFA 中专家的数量。
                - 'smfa_growth' (int): SMFA 中增长率。
                - 'pcfn_growth' (int): PCFN 中增长率。
                - 'snfa_dropout' (float): SMFA 中的 Dropout 概率。
                - 'pcfn_dropout' (float): PCFN 中的 Dropout 概率。
                - 'p_rate' (float): PCFN 中用于分割通道的比例。

    Input:
        x (torch.Tensor): 形状为 [B, 3, H, W] 的输入图像。

    Output:
        torch.Tensor: 形状为 [B, 3, H*upscaling_factor, W*upscaling_factor] 的超分辨率图像。
    """

    def __init__(self, dim=96, n_blocks=8, upscaling_factor=8, fmb_params=None):
        super().__init__()
        self.scale = upscaling_factor  # 上采样因子
        self.to_feat = nn.Conv2d(3, dim, 1, 1, 0)  # 1x1 卷积，用于提取初始特征

        # 设置 FMB 模块的参数
        if fmb_params is None:
            fmb_params = {}  # 使用默认参数

        # 构建 FMB 模块序列
        self.feats = nn.Sequential(
            *[FMB(dim, **fmb_params) for _ in range(n_blocks)]
        )  # FMB 模块的序列

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 1, 1, 0),  # 1x1 卷积
            # mscheadv5(dim, 3 * upscaling_factor**2),
            nn.PixelShuffle(upscaling_factor),  # PixelShuffle 上采样
            nn.Conv2d(3, 3, 3, 1, 1),  # 3x3 卷积，用于调整输出
            # nn.Tanh(),
        )

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入图像。

        Returns:
            torch.Tensor: 超分辨率图像。
        """
        x = self.to_feat(x)
        x = self.feats(x)
        x = self.to_img(x)
        return x


if __name__ == "__main__":
    # 示例 1: 使用默认 FMB 参数
    model1 = SMFANet(dim=96, n_blocks=8, upscaling_factor=8)
    model1.eval()
    dummy_input1 = torch.randn(1, 3, 80, 56)
    with torch.no_grad():
        output1 = model1(dummy_input1)
    print("Example 1 - Input shape:", dummy_input1.shape)
    print("Example 1 - Output shape:", output1.shape)

    # 示例 2: 自定义 FMB 参数
    fmb_custom_params = {
        "smfa_expert": 8,
        "smfa_growth": 2.0,
        "pcfn_growth": 6.0,
        "snfa_dropout": 0.1,
        "pcfn_dropout": 0.2,
        "p_rate": 0.3,
    }
    model2 = SMFANet(
        dim=96, n_blocks=8, upscaling_factor=8, fmb_params=fmb_custom_params
    )
    model2.eval()
    dummy_input2 = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output2 = model2(dummy_input2)
    print("Example 2 - Input shape:", dummy_input2.shape)
    print("Example 2 - Output shape:", output2.shape)
