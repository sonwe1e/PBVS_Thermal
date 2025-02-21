# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   nafnet.py
@Time    :   2023/2/1 20:08
@Desc    :
"""
import torch

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvBNReLU2D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        act=None,
        norm=None,
    ):
        super(ConvBNReLU2D, self).__init__()

        self.layers = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.act = None
        self.norm = None
        if norm == "BN":
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm = torch.nn.InstanceNorm2d(out_channels)
        elif norm == "GN":
            self.norm = torch.nn.GroupNorm(2, out_channels)
        elif norm == "WN":
            self.layers = torch.nn.utils.weight_norm(self.layers)

        if act == "PReLU":
            self.act = torch.nn.PReLU()
        elif act == "SELU":
            self.act = torch.nn.SELU(True)
        elif act == "LeakyReLU":
            self.act = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
        elif act == "ELU":
            self.act = torch.nn.ELU(inplace=True)
        elif act == "ReLU":
            self.act = torch.nn.ReLU(True)
        elif act == "Tanh":
            self.act = torch.nn.Tanh()
        elif act == "Sigmoid":
            self.act = torch.nn.Sigmoid()
        elif act == "SoftMax":
            self.act = torch.nn.Softmax2d()

    def forward(self, inputs):

        out = self.layers(inputs)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class AvgPool2d(nn.Module):
    def __init__(
        self,
        kernel_size=None,
        base_size=None,
        auto_pad=True,
        fast_imp=False,
        train_size=None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return "kernel_size={}, base_size={}, stride={}, fast_imp={}".format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(
                    w - 1, self.kernel_size[1] // r2
                )
                out = (
                    s[:, :, :-k1, :-k2]
                    - s[:, :, :-k1, k2:]
                    - s[:, :, k1:, :-k2]
                    + s[:, :, k1:, k2:]
                ) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = (
                s[:, :, :-k1, :-k2],
                s[:, :, :-k1, k2:],
                s[:, :, k1:, :-k2],
                s[:, :, k1:, k2:],
            )
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode="replicate")

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(
                base_size=base_size, fast_imp=fast_imp, train_size=train_size
            )
            assert m.output_size == 1
            setattr(model, n, pool)


class Local_Base:
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):

        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        # x = x + inp
        # print(x.size(), samples['lr_up'].size())
        # print(x.shape)
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(
        self,
        *args,
        train_size=(1, 4, 256, 256),
        base_size=(640, 640),
        fast_imp=False,
        **kwargs
    ):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        if base_size is None:
            base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


# def get_bare_model(self, net):
#     """Get bare model, especially under wrapping with
#     DistributedDataParallel or DataParallel.
#     """
#     if isinstance(net, (DataParallel, DistributedDataParallel)):
#             net = net.module
#         return net


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if "module." in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


class PBVS(nn.Module):
    def __init__(self, embed_dim):
        super(PBVS, self).__init__()
        self.head = nn.Sequential(
            ConvBNReLU2D(1, out_channels=embed_dim, kernel_size=7, padding=3),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(
                embed_dim, out_channels=embed_dim, kernel_size=3, act="PReLU", padding=1
            ),
        )

    def forward(self, sample):
        return self.head(sample["lr_up"])


class NIR(nn.Module):
    def __init__(self, embed_dim):
        super(NIR, self).__init__()
        self.rgb = nn.Sequential(
            ConvBNReLU2D(3, out_channels=embed_dim, kernel_size=7, padding=3),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(
                embed_dim, out_channels=embed_dim, kernel_size=3, act="PReLU", padding=1
            ),
        )

        self.nir = nn.Sequential(
            ConvBNReLU2D(1, out_channels=embed_dim, kernel_size=7, padding=3),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(
                embed_dim, out_channels=embed_dim, kernel_size=3, act="PReLU", padding=1
            ),
        )
        self.fuse = nn.Sequential(
            ConvBNReLU2D(
                embed_dim * 2 + 4, out_channels=embed_dim, kernel_size=3, padding=1
            ),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(
                embed_dim, out_channels=embed_dim, kernel_size=3, act="PReLU", padding=1
            ),
        )

    def forward(self, sample):
        nir, rgb = sample["lr_up"], sample["img_rgb"]
        # print(nir.shape, rgb.shape)
        out = torch.cat((self.rgb(rgb), self.nir(nir), rgb, nir), dim=1)
        return self.fuse(out)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args

        self.head = PBVS(args.dim)
        self.args = args
        enc_blks = [2, 2, 4, 8]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2]
        train_size = (1, args.dim, args.patch_size, args.patch_size)
        if args.test_only and args.tlc_enhance:
            self.net = NAFNetLocal(
                img_channel=args.dim,
                width=args.dim,
                middle_blk_num=middle_blk_num,
                enc_blk_nums=enc_blks,
                base_size=(640, 480),
                dec_blk_nums=dec_blks,
                train_size=train_size,
            )
        else:
            self.net = NAFNet(
                img_channel=args.dim,
                width=args.dim,
                middle_blk_num=middle_blk_num,
                enc_blk_nums=enc_blks,
                dec_blk_nums=dec_blks,
            )

        self.tail = ConvBNReLU2D(
            in_channels=args.dim, out_channels=1, kernel_size=3, padding=1
        )
        # if args.pre_trained:
        #     load_net = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        #     self.net.load_state_dict(load_net['params'] if 'params' in load_net.keys() else load_net, strict=True)
        #     print('Successfully load pre-trained model ...')

    def forward(self, samples):
        out = self.tail(self.net(self.head(samples)))
        out = out if self.args.no_res else out + samples["lr_up"]
        return {"img_out": out if self.args.test_only else out}


def make_model(args):
    return Net(args)


if __name__ == "__main__":
    img_channel = 3

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    net = NAFNet(img_channel=img_channel)

    inp_shape = (3, 640, 448)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(
        net, inp_shape, verbose=False, print_per_layer_stat=False
    )

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
