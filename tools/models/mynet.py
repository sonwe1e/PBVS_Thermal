import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import LayerNorm2d
from timm.models.layers import DropPath, Mlp
import torch
from thop import profile


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
    def __init__(self, dim, heads=4, dim_head=48, downsample=8):
        super().__init__()
        self.downsample = nn.AvgPool2d(downsample)
        self.attention = OriginAttention(dim, heads, dim_head)
        self.refine = nn.Conv2d(dim, dim, 3, 1, 1)
        self.ffn = Mlp(dim, int(dim * 4), use_conv=True)
        self.ln1 = LayerNorm2d(dim)
        self.ln2 = LayerNorm2d(dim)

    def forward(self, x):
        _, _, h, w = x.shape
        attn_down = self.attention(self.downsample(x))
        x = self.ln1(self.refine(F.interpolate(attn_down, size=(h, w)))) + x
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


class FFTAttention(nn.Module):
    def __init__(self, embed_dim, fft_norm="ortho"):
        super(FFTAttention, self).__init__()
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
        self.atten = FFTAttention(self.atten_channels)
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
        self.lkfb = LKFB(dim, dim, conv=PartialBSConvU)  # 局部关键特征融合
        self.ln_lkfb = LayerNorm2d(dim)
        # self.attn = AttentionFFN(dim, heads=2, dim_head=64, downsample=4)  # 注意力机制

    def forward(self, x):
        x = self.ln_lkfb(self.lkfb(x)) + x
        # x = self.attn(x)
        return x


class FusionNet(nn.Module):
    def __init__(self, dim=64, n_blocks=6, upscaling_factor=4):
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


# class FusionNet(nn.Module):
#     def __init__(self, dim=64, n_blocks=6, upscaling_factor=4):
#         super().__init__()
#         self.scale = upscaling_factor  # 上采样因子
#         self.to_feat = nn.Conv2d(3, dim, 1, 1, 0)

#         self.fusions = nn.ModuleList([Fusion(dim) for _ in range(n_blocks)])

#         self.to_img = nn.Sequential(
#             nn.Conv2d(dim * (n_blocks + 1), 3 * upscaling_factor**2, 1, 1, 0),
#             nn.PixelShuffle(upscaling_factor),  # PixelShuffle 上采样
#             nn.Conv2d(3, 3, 3, 1, 1),  # 3x3 卷积，用于调整输出
#         )
#         self.n_blocks = n_blocks
#         self.dim = dim

#     def forward(self, x):
#         x = self.to_feat(x)
#         features = [x]
#         for fusion in self.fusions:
#             x = fusion(x)
#             features.append(x)
#         x = torch.cat(features, dim=1)  # 拼接所有特征
#         x = self.to_img(x)
#         return x


if __name__ == "__main__":
    # 创建 FusionNet 实例
    model = FusionNet()
    # 创建一个随机输入张量，模拟 256x256 的图像
    dummy_input = torch.randn(1, 3, 123, 256)  # 1 张图片，3 个通道，256x256 大小
    # 使用 thop 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(dummy_input,))
    # 打印结果
    print(f"FLOPs: {flops / 1e9:.4f}G")
    print(f"Number of parameters: {params / 1e6:.4f}M")
