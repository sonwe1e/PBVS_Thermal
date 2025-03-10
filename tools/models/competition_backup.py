import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import LayerNorm2d
from timm.models.layers import DropPath, Mlp
from typing import Optional


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将输入张量划分为窗口。
    参数:
        x (Tensor): 输入特征图，形状为 (B, H, W, C)。
        window_size (int): 窗口的高度和宽度（假定H和W能被window_size整除）。
    返回:
        Tensor: 窗口张量，形状为 (B*num_windows, window_size, window_size, C)。
    """
    B, H, W, C = x.shape  # 获取批大小、高度、宽度、通道数
    # 将每个图像划分成窗口，view的含义如下:
    # (B, H//window_size, window_size, W//window_size, window_size, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 调整维度顺序，将 window_size 维度移到相邻位置，以便之后展平
    # 维度含义调整为: (B, num_windows_H, num_windows_W, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # 合并批次维度和窗口数维度，将每个窗口作为一个独立样本
    windows = x.view(-1, window_size, window_size, C)
    return windows


class WindowAttention(nn.Module):
    r"""窗口多头自注意力模块（Window-based Multi-Head Self-Attention）。
    支持普通窗口和移位窗口机制。
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        """
        参数:
            dim (int): 输入通道数（嵌入维度）。
            window_size (int): 窗口大小（假定窗口为正方形，实际可传入tuple）。
            num_heads (int): 注意力头数。
            qkv_bias (bool): 若为True，为q,k,v映射添加可学习的偏置。默认True。
            qk_scale (float): 若提供，则用于缩放q*k^T的比例因子；否则使用默认 head_dim^-0.5。
            attn_drop (float): 注意力矩阵上的Dropout比率。默认0.0。
            proj_drop (float): 输出上的Dropout比率。默认0.0。
        """
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)  # 窗口的高和宽 (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 如果未指定qk_scale，则使用默认缩放因子 1/sqrt(head_dim)
        self.scale = qk_scale or head_dim**-0.5

        # 相对位置偏差表：大小为 (2*Wh-1)*(2*Ww-1) x num_heads
        # 这是每对位置之间的相对偏差，一共Wh*Ww个token，相对位置索引范围在[-(Wh-1), Wh-1]和[-(Ww-1), Ww-1]之间
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # 计算每个窗口内的相对位置索引，用于在forward中查表获取偏差值
        Wh, Ww = self.window_size
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 形状: (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # 形状: (2, Wh*Ww)
        # 相对坐标差值 (2, Wh*Ww, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # 转置为 (Wh*Ww, Wh*Ww, 2)
        # 将坐标起点移到(0,0)
        relative_coords[..., 0] += Wh - 1  # 横向偏移加上 Wh-1
        relative_coords[..., 1] += Ww - 1  # 纵向偏移加上 Ww-1
        relative_coords[..., 0] *= 2 * Ww - 1  # 合并两个坐标，将它们映射到一维索引
        relative_position_index = relative_coords[..., 0] + relative_coords[..., 1]
        self.register_buffer("relative_position_index", relative_position_index.long())

        # 定义q, k, v的线性映射，以及注意力和输出的Dropout和最终投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # 初始化相对位置偏差表的参数（使用截断正态分布）
        nn.init.trunc_normal_(
            self.relative_position_bias_table, std=0.02
        )  # 若无该函数可改用normal_初始化

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算窗口内的多头自注意力。
        参数:
            x (Tensor): 窗口展平后的输入，形状为 (num_windows*B, N, C)，其中N=window_size*window_size。
            mask (Tensor): 掩码张量，形状 (num_windows, N, N)，在使用移位窗口时用于掩盖无效的注意力连接。没有移位时可为None。
        返回:
            Tensor: 经过窗口自注意力计算的张量，形状与输入 x 相同。
        """

        B_, _, __, C = (
            x.shape
        )  # B_ 是窗口数乘批大小, N 是每个窗口内元素数 (window_size*window_size), C是通道数
        N = self.window_size[0] * self.window_size[1]
        x = x.view(B_, N, C)  # 将每个窗口内的元素展平，形状: [B_, N, C]
        # 1. 计算 q, k, v
        qkv = self.qkv(x)  # [B_, N, 3*C]
        # 将 qkv 拆分为 q, k, v，并reshape为多头的形式
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 现在形状: [3, B_, num_heads, N, head_dim]
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # 分别取出 q, k, v，形状: [B_, num_heads, N, head_dim]

        # 2. 计算注意力分数
        q = q * self.scale  # 缩放查询向量
        attn = q @ k.transpose(
            -2, -1
        )  # 矩阵乘法计算 q 和 k^T 的相关性，形状: [B_, num_heads, N, N]

        # 添加相对位置偏差，每个头都有一组偏差值
        # 根据预先计算的 relative_position_index 从 bias_table 中取出偏差值
        Wh, Ww = self.window_size
        # 获取 shape 为 (N, N, num_heads) 的偏差矩阵
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)  # 将2D索引展平为1D索引
        ]
        relative_position_bias = relative_position_bias.view(
            Wh * Ww, Wh * Ww, -1
        )  # 恢复为 (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # 转为 (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(
            0
        )  # 将偏差添加到注意力得分 (broadcast到batch维度)

        # 如果提供了 mask（移位窗口情况），则加上 mask（无效位置为 -inf，会在softmax时变为0）
        if mask is not None:
            # 假设 mask.shape = (num_windows, N, N)
            nw = mask.shape[0]  # 每个批次中的窗口数
            attn = attn.view(B_ // nw, nw, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # 对应加上mask
            attn = attn.view(-1, self.num_heads, N, N)  # 展平回原来的形状

        # 3. 计算注意力权重并应用 dropout
        attn = self.softmax(attn)  # 对最后两个维度(N, N)做softmax，得到注意力矩阵
        attn = self.attn_drop(attn)  # 可选的dropout

        # 4. 将注意力权重应用到 v 上
        x = attn @ v  # [B_, num_heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B_, N, C)  # 合并多头结果，形状: [B_, N, C]

        # 5. 输出投影
        x = self.proj(x)  # [B_, N, C]
        x = self.proj_drop(x)  # 应用输出dropout
        return x


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """
    将窗口张量恢复为原始特征图。
    参数:
        windows (Tensor): 窗口张量，形状 (B*num_windows, window_size, window_size, C)。
        window_size (int): 窗口大小（高度和宽度）。
        H (int): 原始特征图的高度。
        W (int): 原始特征图的宽度。
    返回:
        Tensor: 恢复后的特征图，形状为 (B, H, W, C)。
    """
    # B 是批大小，计算方法：窗口总数 = (H/window_size)*(W/window_size)*B
    # 因此 B = windows.shape[0] / ((H/window_size)*(W/window_size))
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    # 恢复视图，将张量reshape回 (B, num_windows_H, num_windows_W, window_size, window_size, C)
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    # permute调整回 (B, H//window_size, window_size, W//window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # 最终reshape为 (B, H, W, C)
    x = x.view(B, H, W, -1)
    return x


class WindowAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.wa = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * 4), drop=0.0)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # 调整维度顺序，使通道维度在最后
        x = x + window_reverse(
            self.wa(self.norm1(window_partition(x, self.wa.window_size[0])), mask=mask),
            self.wa.window_size[0],
            h,
            w,
        )
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 3, 1, 2)
        return x


class MultiQueryAttentionV2(nn.Module):
    """Multi Query Attention.

    Fast Transformer Decoding: One Write-Head is All You Need
    https://arxiv.org/pdf/1911.02150.pdf

    This is an acceletor optimized version - removing multiple unnecessary
    tensor transpose by re-arranging indices according to the following rules: 1)
    contracted indices are at the end, 2) other indices have the same order in the
    input and output tensores.

    Compared to V1, this gives 3x speed up.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        num_heads: int = 8,
        key_dim: int = 64,
        value_dim: int = 64,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """Initializer."""
        super().__init__()
        dim_out = dim_out or dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.scale = key_dim**-0.5

        self.query_proj = nn.Parameter(torch.randn([self.num_heads, self.key_dim, dim]))
        self.key_proj = nn.Parameter(torch.randn([dim, self.key_dim]))
        self.value_proj = nn.Parameter(torch.randn([dim, self.value_dim]))
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Parameter(
            torch.randn([dim_out, self.num_heads, self.value_dim])
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def _reshape_input(self, t):
        """Reshapes a tensor to three dimensions, keeping the first and last."""
        s = t.shape
        # Propagate the shape statically where possible.
        # num = t.shape[1:-1].numel()
        # return t.reshape(s[0], num, s[-1])
        return t.reshape(s[0], s[1], -1).transpose(1, 2)

    def forward(self, x, m: Optional[torch.Tensor] = None):
        """Run layer computation."""
        b, _, h, w = x.shape
        m = m if m is not None else x

        reshaped_x = self._reshape_input(x)
        reshaped_m = self._reshape_input(m)

        q = torch.einsum("bnd,hkd->bnhk", reshaped_x, self.query_proj)
        k = torch.einsum("bmd,dk->bmk", reshaped_m, self.key_proj)

        attn = torch.einsum("bnhk,bmk->bnhm", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = torch.einsum("bmd,dv->bmv", reshaped_m, self.value_proj)
        o = torch.einsum("bnhm,bmv->bnhv", attn, v)
        result = torch.einsum("bnhv,dhv->bdn", o, self.out_proj)
        result = self.proj_drop(result)
        return result.reshape(b, -1, h, w)


class FourModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 检查 in_channels 是否能被4整除
        assert in_channels % 4 == 0, "in_channels 必须能被4整除"
        self.c1 = in_channels * 3 // 4  # fea1 的通道数 (3/4)
        self.c2 = in_channels - self.c1  # fea2 的通道数 (1/4)

        self.conv_expand = nn.Conv2d(self.c1, 4 * self.c1, kernel_size=1, bias=False)

        # 分成4份后的分支，各自输入和输出的通道数均为 self.c1
        self.branch1 = nn.Conv2d(self.c1, self.c1, kernel_size=1, bias=True)
        self.branch2 = nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=1, bias=True)
        # 3x3 空洞卷积：为了保持“same”效果，padding = dilation = 2
        self.branch3 = nn.Conv2d(
            self.c1, self.c1, kernel_size=3, padding=2, dilation=2, bias=True
        )
        self.branch4 = nn.Conv2d(self.c1, self.c1, kernel_size=7, padding=3, bias=True)

        # 1x1 卷积：将4个分支拼接后的通道数（4*self.c1）降回 self.c1
        self.conv_reduce = nn.Conv2d(4 * self.c1, self.c1, kernel_size=1, bias=True)

    def forward(self, x):
        # x: [B, in_channels, H, W]
        # 将 x 分为 fea1 和 fea2
        fea1 = x[:, : self.c1, :, :]  # 前3/4通道
        fea2 = x[:, self.c1 :, :, :]  # 后1/4通道

        # 对 fea1 进行通道扩展：1x1卷积扩展至4倍通道数
        expanded = self.conv_expand(fea1)  # [B, 4*self.c1, H, W]

        # 将扩展后的特征均分为4份
        splits = torch.chunk(expanded, 4, dim=1)  # 每个元素形状为 [B, self.c1, H, W]

        out1 = self.branch1(splits[0])
        out2 = self.branch2(splits[1])
        out3 = self.branch3(splits[2])
        out4 = self.branch4(splits[3])

        # 拼接4个分支的输出，得到 [B, 4*self.c1, H, W]
        concatenated = torch.cat([out1, out2, out3, out4], dim=1)

        # 通过 1x1 卷积降维回 fea1 原来的通道数
        reduced = self.conv_reduce(concatenated)  # [B, self.c1, H, W]

        # 最后将处理后的 fea1 与原始的 fea2 拼接，恢复到 in_channels
        out = torch.cat([reduced, fea2], dim=1)  # [B, in_channels, H, W]
        return out


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, self.dim_head, h * w), qkv)

        q = q.softmax(dim=-2)  # 特征维度归一化
        k = k.softmax(dim=-1)  # 空间维度归一化

        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.view(b, -1, h, w)
        return self.to_out(out)


class OriginAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        # 定义卷积层，用于生成 q, k, v
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        # 定义输出卷积层
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # 获取 q, k, v，通过卷积生成
        qkv = self.to_qkv(x).reshape(b, 3, self.heads, self.dim_head, h, w)
        q, k, v = qkv.unbind(dim=1)

        # 对 q, k, v 进行调整，将 h, w 维度展平
        q = q.view(b, self.heads, self.dim_head, -1).transpose(2, 3)
        k = k.view(b, self.heads, self.dim_head, -1).transpose(2, 3)
        v = v.view(b, self.heads, self.dim_head, -1).transpose(2, 3)
        # 计算注意力矩阵
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, heads, h*w, h*w]
        # attn = attn.softmax(dim=-1)  # 注意力归一化
        attn = F.sigmoid(attn)  # 注意力归一化

        # 加权求和得到输出
        out = attn @ v  # [b, heads, h*w, dim_head]
        out = out.transpose(1, 2).reshape(b, self.heads * self.dim_head, h, w)

        # 通过 to_out 卷积层进行最终映射
        x = self.to_out(out)
        return x


class AttentionFFN(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, downsample=8):
        super().__init__()
        self.downsample = nn.AvgPool2d(downsample)
        self.attention = OriginAttention(dim, heads, dim_head)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=downsample, mode="bicubic", align_corners=False),
            nn.Conv2d(dim, dim, 3, 1, 1),  # 上采样后卷积增强细节
        )
        self.ffn = Mlp(dim, int(dim * 4), use_conv=True, drop=0.12)
        self.ln1 = LayerNorm2d(dim)
        self.ln2 = LayerNorm2d(dim)

    def forward(self, x):
        x = self.ln1(self.upsample(self.attention(self.downsample(x)))) + x
        x = self.ln2(self.ffn(x)) + x
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
            mode="bicubic",
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
        self.attn = AttentionFFN(dim)  # 动态多层感知机
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
        x = self.attn(x)
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
        x = self.feats(x) + x
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
