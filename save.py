import os
import argparse
import numpy as np
from PIL import Image
import lpips
import torch
from skimage.metrics import structural_similarity as ssim


def load_and_process_image(path, target_size=(256, 256)):
    """加载图像并处理为SSIM计算和LPIPS计算需要的格式"""
    img = Image.open(path).convert("L")  # 转为灰度
    img = img.resize(target_size)

    # 处理SSIM需要的格式
    img_np = np.array(img).astype(np.float32)

    # 处理LPIPS需要的格式（转换为伪RGB）
    img_rgb = np.stack([img_np] * 3, axis=-1)  # 复制为三通道
    img_rgb = (img_rgb / 255.0) * 2 - 1  # 归一化到[-1,1]
    img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float()

    return img_np, img_tensor


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device)

    # 加载img2图像
    img2_dir = args.img2
    img2_paths = [
        os.path.join(img2_dir, f)
        for f in sorted(os.listdir(img2_dir))
        if f.lower().endswith(".bmp")
    ]
    assert len(img2_paths) == 20, "img2目录应包含20张.bmp图像"

    img2_ssim = []
    img2_lpips = []
    for path in img2_paths:
        ssim_img, lpips_img = load_and_process_image(path, (args.size, args.size))
        img2_ssim.append(ssim_img)
        img2_lpips.append(lpips_img.to(device))

    # 加载img1图像
    img1_dir = args.img1
    img1_paths = [
        os.path.join(img1_dir, f)
        for f in sorted(os.listdir(img1_dir))
        if f.lower().endswith(".bmp")
    ]

    scores = []
    for img1_path in img1_paths:
        # 处理当前img1图像
        ssim_img1, lpips_img1 = load_and_process_image(
            img1_path, (args.size, args.size)
        )
        lpips_img1 = lpips_img1.to(device)

        # 计算指标
        total_ssim = 0.0
        total_lpips = 0.0
        for ssim_img2, lpips_img2 in zip(img2_ssim, img2_lpips):
            # 计算SSIM
            total_ssim += ssim(ssim_img1, ssim_img2, data_range=255)

            # 计算LPIPS
            with torch.no_grad():
                total_lpips += loss_fn(lpips_img1, lpips_img2).item()

        # 计算平均分
        avg_ssim = total_ssim / 20
        avg_lpips = total_lpips / 20
        scores.append((img1_path, avg_ssim, avg_lpips))

    # 标准化分数
    ssim_scores = np.array([s[1] for s in scores])
    lpips_scores = np.array([s[2] for s in scores])

    # SSIM越高越好，LPIPS越低越好
    norm_ssim = (ssim_scores - ssim_scores.min()) / (
        ssim_scores.max() - ssim_scores.min()
    )
    norm_lpips = 1 - (lpips_scores - lpips_scores.min()) / (
        lpips_scores.max() - lpips_scores.min()
    )

    # 综合得分（各占50%权重）
    combined_scores = 0.5 * norm_ssim + 0.5 * norm_lpips

    # 排序并选择前n个
    sorted_indices = np.argsort(combined_scores)[::-1]  # 降序排列
    selected_indices = sorted_indices[: args.n]

    # 输出结果
    print(f"Selected top {args.n} images from img1:")
    for idx in selected_indices:
        print(f"Path: {scores[idx][0]}")
        print(f"SSIM: {scores[idx][1]:.4f}, LPIPS: {scores[idx][2]:.4f}")
        print("-" * 50)

    # 如果需要保存到指定目录
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        for idx in selected_indices:
            img = Image.open(scores[idx][0])
            save_path = os.path.join(args.output, os.path.basename(scores[idx][0]))
            img.save(save_path)
        print(f"Selected images saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select images from img1 that are similar to img2"
    )
    parser.add_argument(
        "--img1",
        type=str,
        default="/media/hdd/sonwe1e/Competition/PBVS_Thermal/Data/train/GT",
    )
    parser.add_argument(
        "--img2",
        type=str,
        default="/media/hdd/sonwe1e/Competition/PBVS_Thermal/Data/valid/GT",
    )
    parser.add_argument("--n", type=int, help="Number of images to select", default=100)
    parser.add_argument(
        "--output",
        type=str,
        default="/media/hdd/sonwe1e/Competition/PBVS_Thermal/Data/train/sim",
    )
    parser.add_argument("--size", type=int, default=256, help="Image processing size")
    args = parser.parse_args()

    main(args)
