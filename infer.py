import torch
from tools.utils import load_model, infer_from_model
from tools.models.mynet import FusionNet
from tools.models.lkfn import LKFN
from tools.models.efdn import EFDN
from tools.utils import analyze_data, display_top_bottom, plot_histograms
import matplotlib.pyplot as plt


model = FusionNet(32, 8, 4)
# model = LKFN()
# model = EFDN()
# model.load_state_dict(torch.load("checkpoints/team00_EFDN.pth"))
model_path = "checkpoints/v2_cat_test/epoch_143-loss_23.327.ckpt"
images_path = "DF2K/test_phase1"
output_path = "DF2K/pred_phase1"

# 加载模型
# model = load_model(model, model_path)

# 进行推理, and pass output_path
infer_from_model(model, images_path, output_path, tta=False, num_workers=16)
# Global settings for plot appearance
plt.rcParams["figure.figsize"] = (12, 12)  # Adjust as needed
plt.rcParams["figure.dpi"] = 300


gt_path = "DF2K/valid"
pred_path = "DF2K/pred_phase1"
num_cols = 4
valid_results, psnr_values, ssim_values = analyze_data(gt_path, pred_path)
# --- Display Top/Bottom Images ---
display_top_bottom(valid_results, "PSNR", num_cols, reverse_sort=True)  # Top
display_top_bottom(valid_results, "PSNR", num_cols, reverse_sort=False)  # Down


# --- Histograms ---
plot_histograms(psnr_values, ssim_values)
