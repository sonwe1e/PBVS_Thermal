import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import LayerNorm2d
from timm.models.layers import DropPath, Mlp


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        # 线性变换生成Q、K、V
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, bias=False)
        # 输出变换
        self.to_out = nn.Conv2d(inner_dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        # 生成Q、K、V [b, 3*inner_dim, h, w]
        qkv = self.to_qkv(x)
        q, k, v = torch.split(qkv, self.heads * self.dim_head, dim=1)

        # 拆分为多头 [b, heads, dim_head, h*w]
        q = q.view(b, self.heads, self.dim_head, h * w)
        k = k.view(b, self.heads, self.dim_head, h * w)
        v = v.view(b, self.heads, self.dim_head, h * w)

        # 线性注意力核心计算
        q = q.softmax(dim=-2)  # 在特征维度做softmax
        k = k.softmax(dim=-1)  # 在空间维度做softmax

        # 计算上下文向量 [b, heads, dim_head, dim_head]
        context = torch.einsum("bhdn,bhen->bhde", k, v)

        # 聚合上下文 [b, heads, dim_head, h*w]
        out = torch.einsum("bhde,bhdn->bhen", context, q)

        # 合并多头 [b, heads*dim_head, h, w]
        out = out.view(b, self.heads * self.dim_head, h, w)

        # 输出变换
        return self.to_out(out)


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = LinearAttention(dim, heads=num_heads)
        self.drop_path = DropPath(drop) if drop > 0.0 else nn.Identity()
        self.norm2 = LayerNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=drop,
            use_conv=True,
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.drop_path(self.attn(x)) + identity
        identity = x
        x = self.norm2(x)
        x = self.drop_path(self.mlp(x)) + identity
        return x


class STLConv(nn.Module):
    def __init__(self, dim, out_dim, kernel_size=3, stride=1, padding=1):
        super(STLConv, self).__init__()
        self.stl = TransformerLayer(dim)
        self.conv = nn.Conv2d(dim, out_dim, kernel_size, stride=stride, padding=padding)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.stl(x)
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x


class ResidualDenseGroup(nn.Module):
    def __init__(self, dim):
        super(ResidualDenseGroup, self).__init__()
        self.l1 = STLConv(dim, dim // 2, kernel_size=3, stride=1, padding=1)
        self.l2 = STLConv(dim * 3 // 2, dim // 2, kernel_size=3, stride=1, padding=1)
        self.l3 = STLConv(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        self.fusion = STLConv(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.l1(x)  # c // 4
        x2 = self.l2(torch.cat([x, x1], dim=1))  # 5c // 4
        x3 = self.l3(torch.cat([x, x1, x2], dim=1))  # 3c // 2
        x = self.fusion(x3)  # c
        return x


class BSConvU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class PartialBSConvU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        scale=2,
    ):
        super().__init__()

        # pointwise
        self.remaining_channels = in_channels // scale
        self.other_channels = in_channels - self.remaining_channels
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # partialdepthwise
        self.pdw = nn.Conv2d(
            in_channels=self.remaining_channels,
            out_channels=self.remaining_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels // scale,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea1, fea2 = torch.split(
            fea, [self.remaining_channels, self.other_channels], dim=1
        )
        fea1 = self.pdw(fea1)
        fea = torch.cat((fea1, fea2), 1)
        fea = self.pw(fea)
        return fea


class Attention(nn.Module):
    def __init__(self, embed_dim, fft_norm="ortho"):
        super(Attention, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv_layer2 = torch.nn.Conv2d(embed_dim // 2, embed_dim // 2, 1, 1, 0)
        self.conv_layer3 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        real = ffted.real + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.real))))
        )
        imag = ffted.imag + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.imag))))
        )

        ffted = torch.complex(real, imag)
        ifft_shape_slice = x.shape[-2:]
        atten = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )

        return x * atten


class LKFB(nn.Module):
    def __init__(self, in_channels, out_channels, atten_channels=None, conv=BSConvU):
        super().__init__()

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        if atten_channels is None:
            self.atten_channels = in_channels
        else:
            self.atten_channels = atten_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=5, padding=2)
        self.c2_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c2_r = conv(self.rc, self.rc, kernel_size=5, padding=2)
        self.c3_d = nn.Conv2d(self.rc, self.dc, 1)
        self.c3_r = conv(self.rc, self.rc, kernel_size=5, padding=6, dilation=3)

        self.c4 = BSConvU(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, self.atten_channels, 1, 1, 0)
        self.atten = Attention(self.atten_channels)
        self.c6 = nn.Conv2d(self.atten_channels, out_channels, 1, 1, 0)
        # self.pixel_norm = nn.LayerNorm(out_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)

        out = self.atten(out)
        out = self.c6(out)

        # out = out.permute(0, 2, 3, 1)
        # out = self.pixel_norm(out)
        # out = out.permute(0, 3, 1, 2).contiguous()

        return out


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
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))  # 只对部分通道进行卷积
            x = self.conv_2(self.dp(torch.cat([x1, x2], dim=1)))  # 合并通道
        else:  # 推理模式
            x = self.act(self.conv_0(x))
            x[:, : self.p_dim, :, :] = self.act(self.conv_1(x[:, : self.p_dim, :, :]))
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

    def __init__(self, dim=36, growth_rate=4, dropout_rate=0.0):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)  # 1x1 卷积，用于增加通道数
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)  # 1x1 卷积
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)  # 1x1 卷积
        self.dp = nn.Dropout(dropout_rate)  # Dropout 层

        self.lde = DMlp(dim, growth_rate, dropout_rate)  # 超分辨率混合专家特征提取器
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)  # 深度可分离卷积

        self.gelu = nn.GELU()  # GELU 激活函数
        self.down_scale = 8  # 降采样比例

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

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


class Fusion(nn.Module):
    def __init__(
        self,
        dim,
        smfa_growth=4,
        pcfn_growth=4,
        snfa_dropout=0.0,
        pcfn_dropout=0.0,
        p_rate=0.25,
    ):
        super().__init__()
        self.smfa = SMFA(dim, smfa_growth, snfa_dropout)
        self.pcfn = PCFN(dim, pcfn_growth, p_rate, pcfn_dropout)  # 部分连接前馈网络
        self.lkfb = LKFB(dim, dim, conv=PartialBSConvU)  # 局部关键特征融合
        self.ln1 = LayerNorm2d(dim)  # Layer Normalization
        self.ln2 = LayerNorm2d(dim)  # Layer Normalization
        self.ln3 = LayerNorm2d(dim)  # Layer Normalization

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入特征图。

        Returns:
            torch.Tensor: 输出特征图。
        """
        x = self.ln1(self.smfa(x)) + x
        x = self.ln2(self.pcfn(x)) + x
        x = self.ln3(self.lkfb(x)) + x
        return x


class FusionNet(nn.Module):
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
            *[Fusion(dim, **fmb_params) for _ in range(n_blocks)]
        )  # FMB 模块的序列

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 1, 1, 0),  # 1x1 卷积
            nn.PixelShuffle(upscaling_factor),  # PixelShuffle 上采样
            nn.Conv2d(3, 3, 3, 1, 1),  # 3x3 卷积，用于调整输出
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


class FusionNetv2(nn.Module):
    """
    Spatial Modulation and Feature Aggregation Network (SMFANet) 模型。
    """

    def __init__(self, dim=96, n_blocks=8, upscaling_factor=8, fmb_params=None):
        super().__init__()
        self.scale = upscaling_factor  # 上采样因子
        self.to_feat = BSConvU(3, dim, kernel_size=3, padding=1)

        # 设置 FMB 模块的参数
        if fmb_params is None:
            fmb_params = {}  # 使用默认参数

        # 构建 FMB 模块序列
        self.feats = nn.ModuleList(
            [Fusion(dim, **fmb_params) for _ in range(n_blocks)]
        )  # FMB 模块的序列

        self.before_img = nn.Sequential(
            nn.Conv2d(dim * n_blocks, dim, 1, 1, 0),  # 1x1 卷积
            nn.GELU(),  # GELU 激活函数
            BSConvU(dim, dim, kernel_size=3, padding=1),  # 3x3 卷积
        )
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),  # 3x3 卷积
            nn.PixelShuffle(upscaling_factor),  # PixelShuffle 上采样
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
        feat = x
        features = []

        # 获取每一层的特征并拼接
        for fmb in self.feats:
            x = fmb(x)
            features.append(x)

        # 将所有特征拼接在一起
        x = torch.cat(features, dim=1)  # 沿着通道维度拼接
        x = self.before_img(x) + feat
        x = self.to_img(x)
        return x


if __name__ == "__main__":
    # 示例 1: 使用默认 FMB 参数
    model1 = FusionNetv2(dim=96, n_blocks=8, upscaling_factor=8)
    model1.eval()
    dummy_input1 = torch.randn(1, 3, 80, 56)
    with torch.no_grad():
        output1 = model1(dummy_input1)
    print("Example 1 - Input shape:", dummy_input1.shape)
    print("Example 1 - Output shape:", output1.shape)

    # 示例 2: 自定义 FMB 参数
    fmb_custom_params = {
        "smfa_growth": 2.0,
        "pcfn_growth": 6.0,
        "snfa_dropout": 0.1,
        "pcfn_dropout": 0.2,
        "p_rate": 0.3,
    }
    model2 = FusionNetv2(
        dim=96, n_blocks=8, upscaling_factor=8, fmb_params=fmb_custom_params
    )
    model2.eval()
    dummy_input2 = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output2 = model2(dummy_input2)
    print("Example 2 - Input shape:", dummy_input2.shape)
    print("Example 2 - Output shape:", output2.shape)
