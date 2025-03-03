import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import LayerNorm2d
from timm.models.layers import DropPath, Mlp
import torch
from thop import profile
from typing import Optional


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
        # attn = attn.softmax(dim=-1)
        attn = F.sigmoid(attn)
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
    def __init__(self, dim, heads=8, dim_head=96, downsample=4):
        super().__init__()
        self.downsample = nn.AvgPool2d(downsample)
        self.attention = OriginAttention(dim, heads, dim_head)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=downsample, mode="bilinear", align_corners=False),
            nn.Conv2d(dim, dim, 3, padding=1),  # 上采样后卷积增强细节
        )
        self.ffn = Mlp(dim, int(dim * 4), use_conv=True)
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


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=4.0, dropout_date=0.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)  # 1x1 卷积，用于增加通道数
        self.act = nn.GELU()  # GELU 激活函数
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)  # 1x1 卷积，用于恢复通道数
        self.dp = nn.Dropout(dropout_date)  # Dropout 层

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.dp(x)
        x = self.conv_1(x)
        return x


class PCFN(nn.Module):
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
    def __init__(self, dim):
        super().__init__()
        self.lkfb = LKFB(dim, dim, conv=BSConvU)  # 局部关键特征融合
        self.ln_lkfb = LayerNorm2d(dim)
        self.attn = AttentionFFN(dim, heads=4, dim_head=64, downsample=8)  # 注意力机制

    def forward(self, x):
        x = self.ln_lkfb(self.lkfb(x)) + x
        x = self.attn(x)
        return x


class FusionNet(nn.Module):
    def __init__(self, dim=96, n_blocks=8, upscaling_factor=8):
        super().__init__()
        self.scale = upscaling_factor  # 上采样因子
        self.to_feat = nn.Conv2d(3, dim, 1, 1, 0)

        self.feats = nn.Sequential(*[Fusion(dim) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 1, 1, 0),  # 1x1 卷积
            nn.PixelShuffle(upscaling_factor),  # PixelShuffle 上采样
            nn.Conv2d(3, 3, 3, 1, 1),  # 3x3 卷积，用于调整输出
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x)
        x = self.to_img(x)
        return x


class Ensemble(nn.Module):
    def __init__(self, model1_cls=FusionNet, model2_cls=FusionNet):  # 假设模型类已定义
        super(Ensemble, self).__init__()
        self.model1 = model1_cls()
        self.model2 = model2_cls()

    def forward(self, x):
        B = x.shape[0]

        # --- Model 1 (翻转 TTA) ---
        x_flip = torch.cat(
            [
                x,
                torch.flip(x, dims=[-1]),  # 水平翻转
                torch.flip(x, dims=[-2]),  # 垂直翻转
                torch.flip(x, dims=[-1, -2]),  # 水平+垂直翻转
            ],
            dim=0,
        )  # shape: (4*B, C, H, W)
        out_flip = self.model1(x_flip)
        # 将4*B的输出reshape为 (4, B, C, H, W)，并进行逆变换
        out_flip = out_flip.view(4, B, *out_flip.shape[1:])
        out_original = out_flip[0]  # 原始图像的输出
        out_hflip = torch.flip(out_flip[1], dims=[-1])  # 水平翻转的逆变换
        out_vflip = torch.flip(out_flip[2], dims=[-2])  # 垂直翻转的逆变换
        out_hvflip = torch.flip(out_flip[3], dims=[-1, -2])  # 水平+垂直翻转的逆变换

        out_flip = torch.stack(
            [out_original, out_hflip, out_vflip, out_hvflip], dim=0
        ).mean(dim=0)  # 逆变换后取平均

        # --- Model 2 (旋转 TTA) ---
        x_rot = torch.cat(
            [
                torch.rot90(x, 1, dims=[-2, -1]),  # 顺时针90度
                torch.rot90(x, -1, dims=[-2, -1]),  # 逆时针90度
            ],
            dim=0,
        )
        out_rot = self.model2(x_rot)
        # 将输出reshape为 (4, B, C, H, W)，并进行逆旋转
        out_rot = out_rot.view(2, B, *out_rot.shape[1:])

        out_rot90 = torch.rot90(out_rot[0], -1, dims=[-2, -1])  # 顺时针90度的逆旋转
        out_rot270 = torch.rot90(out_rot[1], 1, dims=[-2, -1])  # 逆时针90度的逆旋转

        out_rot = torch.stack([out_original, out_rot90, out_rot270], dim=0).mean(dim=0)

        # --- 检查并融合输出 ---
        if out_flip.shape != out_rot.shape:
            raise ValueError(
                f"Model 1 and Model 2 output shapes are incompatible: {out_flip.shape} vs {out_rot.shape}"
            )

        return (out_flip + out_rot) / 2.0  # 或者其他融合策略，如加权平均


if __name__ == "__main__":
    # 创建 FusionNet 实例
    model = FusionNet()
    # 创建一个随机输入张量，模拟 256x256 的图像
    dummy_input = torch.randn(1, 3, 256, 256)  # 1 张图片，3 个通道，256x256 大小
    # 使用 thop 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(dummy_input,))
    # 打印结果
    print(f"FLOPs: {flops / 1e9:.4f}G")
    print(f"Number of parameters: {params / 1e6:.4f}M")
