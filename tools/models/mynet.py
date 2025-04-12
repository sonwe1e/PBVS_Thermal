import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    def __init__(self, dim, dim_scale=4):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim * dim_scale, 1, 1, 0)
        self.conv2 = nn.Conv2d(
            dim * dim_scale, dim * dim_scale, 3, 1, 1, groups=dim * dim_scale
        )
        self.conv3 = nn.Conv2d(dim * dim_scale, dim, 1, 1, 0)

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x + identity


class MYNET(nn.Module):
    def __init__(self, dim=64, n_blocks=12, dim_scale=4, upscaling_factor=2):
        super().__init__()
        self.scale = upscaling_factor
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(
            *[BottleNeck(dim, dim_scale) for _ in range(n_blocks)]
        )
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor),
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x


if __name__ == "__main__":
    model = MYNET()
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(y.shape)  # Should be (1, 3, 128, 128)
