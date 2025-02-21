import torch
import math
import timm
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import LayerNorm2d


class PNet(nn.Module):
    def __init__(self, upscaling_factor=8):
        super().__init__()
        self.scale = upscaling_factor
        self.to_feat = nn.Conv2d(3, 384, 1, 1, 0)
        self.feats = timm.create_model(
            "convnext_large.fb_in22k_ft_in1k_384", pretrained=True
        ).stages[2]
        self.feats.downsample[1] = torch.nn.Conv2d(384, 768, 1, 1, 0)
        self.to_img = nn.Sequential(
            nn.Conv2d(768, 3 * upscaling_factor**2, 1, 1, 0),
            nn.PixelShuffle(upscaling_factor),
            nn.Conv2d(3, 3, 3, 1, 1),
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x)
        x = self.to_img(x)
        return x


if __name__ == "__main__":

    # Create the model and set to evaluation mode
    model = PNet(upscaling_factor=8)
    model.eval()

    # Create a dummy input tensor with batch size 1 and spatial dimensions 64x64
    dummy_input = torch.randn(1, 3, 80, 56)

    # Run the model to obtain the output
    with torch.no_grad():
        output = model(dummy_input)

    # Print input and output shapes
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
