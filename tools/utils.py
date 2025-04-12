import glob
import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import concurrent.futures  # For multithreading
import cv2  # Use OpenCV for faster saving
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
import torch.nn.functional as F


def load_model_from_config(config):
    """根据配置动态加载模型

    Args:
        config: 包含模型配置的对象或字典，例如 {'arch': 'mynet', 'upscaling_factor': 2, 'dim': 32, 'n_blocks': 8}

    Returns:
        加载好的模型实例
    """
    try:
        # 从嵌套配置中获取模型相关参数

        model_config = getattr(config, "model")
        arch = model_config["arch"]

        # 动态导入模块，模块名为小写形式的 arch
        module_name = f"tools.models.{arch.lower()}"
        module = __import__(module_name, fromlist=[arch])

        # 获取模型类
        model_class = getattr(module, arch)

        # 获取模型构造函数的参数签名
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}  # 排除 'self'

        # 动态获取模型配置中的参数，并过滤出模型支持的部分
        if isinstance(model_config, dict):
            kwargs = {
                key: value
                for key, value in model_config.items()
                if key in valid_params and key != "arch"
            }
        else:
            kwargs = {
                key: getattr(model_config, key)
                for key in vars(model_config)
                if key in valid_params and key != "arch"
            }

        # 实例化模型类，只传递有效参数
        model = model_class(**kwargs)

        return model
    except ImportError:
        raise ValueError(f"不支持的模型架构或模块导入失败: {arch}")
    except TypeError as e:
        raise RuntimeError(f"模型 {arch} 实例化失败，可能是参数不匹配: {e}") from e
    except Exception as e:
        raise RuntimeError(f"加载模型 {arch} 失败: {e}") from e


def load_model(model, model_path):
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    # 使用字典推导式一次性处理state_dict
    filtered_dict = {
        k.replace("model.", ""): v for k, v in state_dict.items() if "model" in k
    }
    model.load_state_dict(filtered_dict)
    model.eval()
    return model


def save_image_cv2(img_array, output_path):
    """Saves an image using OpenCV."""
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    cv2.imwrite(output_path, img_array)


def infer_from_model(model, images_path, output_path, tta=False, device="cuda:0"):
    model = model.to(device)
    model.eval()
    if os.path.isdir(images_path):
        image_files = sorted(
            glob.glob(os.path.join(images_path, "*.png"))
            + glob.glob(os.path.join(images_path, "*.jpg"))
        )
    else:
        image_files = [images_path]
    os.makedirs(output_path, exist_ok=True)

    for img_path in tqdm(image_files, desc="Inferencing"):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            if tta:
                # 原始图像及其旋转版本（0度、90度、180度、270度）
                base_imgs = [img_tensor]  # 原始图像（无旋转）
                base_imgs.extend(
                    torch.rot90(img_tensor, k=i, dims=(2, 3)) for i in range(1, 4)
                )

                # 创建所有基础图像的水平翻转（dims=3）和垂直翻转（dims=2）
                hor_flipped_imgs = [torch.flip(img, dims=(3,)) for img in base_imgs]

                # 合并所有增强图像（原始+旋转+水平翻转+垂直翻转）
                tta_imgs = base_imgs + hor_flipped_imgs

                # 获取模型对所有增强图像的预测
                tta_outputs = [model(img) for img in tta_imgs]

                # 初始化输出张量
                output = torch.zeros_like(tta_outputs[0])

                # 处理原始和旋转输出（前4个）
                for i, out in enumerate(tta_outputs[:4]):
                    output += torch.rot90(out, k=-i, dims=(2, 3))

                # 处理水平翻转输出（中间4个）
                for i, out in enumerate(tta_outputs[4:8]):
                    output += torch.rot90(torch.flip(out, dims=(3,)), k=-i, dims=(2, 3))

                output /= len(tta_outputs)
            else:
                output = model(img_tensor)

        # 检查输出范围
        # print(f"Output range: {output.min():.2f} - {output.max():.2f}")
        output = (output[0] * 255.0).cpu().numpy().round()
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))

        img_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        output_file_path = os.path.join(output_path, img_name)
        save_image_cv2(output, output_file_path)


def patch_infer_from_model(
    model,
    images_path,
    output_path,
    patch_size=256,
    overlap=128,
    scale=2,
    tta=True,
    batch_size=1,
    device="cuda:0",
):
    # --- Basic Setup ---
    if batch_size > 1:
        print(
            "Warning: batch_size > 1 is not fully implemented for simplicity with overlap/TTA. Using batch_size=1."
        )
        batch_size = 1

    if overlap >= patch_size:
        raise ValueError("Overlap must be less than patch_size.")

    model = model.to(device)
    model.eval()

    # --- Image Discovery ---
    if os.path.isdir(images_path):
        image_files = sorted(
            glob.glob(os.path.join(images_path, "*.png"))
            + glob.glob(os.path.join(images_path, "*.jpg"))
            + glob.glob(os.path.join(images_path, "*.bmp"))
        )
    elif os.path.isfile(images_path):
        image_files = [images_path]
    else:
        print(f"Error: Input path {images_path} is not a valid file or directory.")
        return

    os.makedirs(output_path, exist_ok=True)

    # --- Main Inference Loop ---
    for img_path in tqdm(image_files, desc="Patch Inferencing"):
        try:
            # --- Image Loading and Preprocessing ---
            img_lq = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_lq is None:
                print(f"Warning: Failed to load image {img_path}. Skipping.")
                continue

            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
            img_lq = img_lq.astype(np.float32) / 255.0

            img_lq_tensor = (
                torch.from_numpy(img_lq).permute(2, 0, 1).unsqueeze(0).to(device)
            )
            b, c, h_lq, w_lq = img_lq_tensor.shape

            # --- Padding Calculation ---
            stride = patch_size - overlap

            # Calculate padding amounts needed so that image dimensions minus patch_size
            # become divisible by stride after padding.
            # Handle cases where h_lq or w_lq might be smaller than patch_size.
            target_h = h_lq
            if h_lq < patch_size:
                target_h = patch_size  # Ensure padded height is at least patch_size
            else:
                pad_needed_h = (stride - (h_lq - patch_size) % stride) % stride
                target_h = h_lq + pad_needed_h

            target_w = w_lq
            if w_lq < patch_size:
                target_w = patch_size  # Ensure padded width is at least patch_size
            else:
                pad_needed_w = (stride - (w_lq - patch_size) % stride) % stride
                target_w = w_lq + pad_needed_w

            pad_h = target_h - h_lq  # Total padding for height
            pad_w = target_w - w_lq  # Total padding for width

            # --- Choose Padding Mode ---
            # If the original image dimension is smaller than the padding required for 'reflect',
            # 'reflect' mode will fail. In such cases, switch to 'replicate'.
            # Check if bottom padding exceeds original height or right padding exceeds original width
            if pad_h >= h_lq or pad_w >= w_lq:
                padding_mode = "replicate"  # Safer mode for small images
                # print(f"Info: Image {os.path.basename(img_path)} smaller than patch/padding. Using '{padding_mode}' mode.") # Optional info
            else:
                padding_mode = "reflect"  # Preferred mode when possible

            # --- Apply Padding ---
            # Pad format is (pad_left, pad_right, pad_top, pad_bottom)
            # We only pad right and bottom here.
            img_lq_padded = F.pad(
                img_lq_tensor, (0, pad_w, 0, pad_h), mode=padding_mode
            )
            b, c, h_pad, w_pad = img_lq_padded.shape
            # print(f"Original size: ({h_lq}, {w_lq}), Padded size: ({h_pad}, {w_pad}), Mode: {padding_mode}") # Debug padding

            # --- Prepare Output Canvas and Weight Map ---
            output_canvas = torch.zeros(
                (b, c, h_pad * scale, w_pad * scale), dtype=torch.float32
            ).to(device)
            weight_map = torch.zeros(
                (b, c, h_pad * scale, w_pad * scale), dtype=torch.float32
            ).to(device)

            # --- Patch Processing ---
            with torch.no_grad():
                for y in range(0, h_pad - patch_size + 1, stride):
                    for x in range(0, w_pad - patch_size + 1, stride):
                        input_patch = img_lq_padded[
                            :, :, y : y + patch_size, x : x + patch_size
                        ]

                        if tta:
                            # TTA Logic (same as before)
                            base_patches = [input_patch]
                            base_patches.extend(
                                torch.rot90(input_patch, k=i, dims=(2, 3))
                                for i in range(1, 4)
                            )
                            flipped_patches = [
                                torch.flip(p, dims=(3,)) for p in base_patches
                            ]
                            tta_patches = base_patches + flipped_patches
                            tta_outputs_patch = [model(p) for p in tta_patches]
                            patch_output_final = torch.zeros_like(tta_outputs_patch[0])
                            for i, out_p in enumerate(tta_outputs_patch[:4]):
                                patch_output_final += torch.rot90(
                                    out_p, k=-i, dims=(2, 3)
                                )
                            for i, out_p in enumerate(tta_outputs_patch[4:]):
                                patch_output_final += torch.rot90(
                                    torch.flip(out_p, dims=(3,)), k=-i, dims=(2, 3)
                                )
                            patch_output_final /= len(tta_outputs_patch)
                        else:
                            patch_output_final = model(input_patch)

                        out_y = y * scale
                        out_x = x * scale
                        out_patch_h, out_patch_w = (
                            patch_output_final.shape[2],
                            patch_output_final.shape[3],
                        )

                        output_canvas[
                            :,
                            :,
                            out_y : out_y + out_patch_h,
                            out_x : out_x + out_patch_w,
                        ] += patch_output_final
                        weight_map[
                            :,
                            :,
                            out_y : out_y + out_patch_h,
                            out_x : out_x + out_patch_w,
                        ] += 1

                output_canvas /= torch.clamp(weight_map, min=1e-8)

            # --- Cropping and Post-processing ---
            output_final = output_canvas[:, :, 0 : h_lq * scale, 0 : w_lq * scale]
            output_final = output_final.squeeze(0).cpu().numpy()
            output_final = np.transpose(output_final, (1, 2, 0))
            output_final = (
                np.clip(output_final * 255.0, 0, 255).round().astype(np.uint8)
            )

            # --- Save Output Image ---
            img_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
            output_file_path = os.path.join(output_path, img_name)
            save_image_cv2(output_final, output_file_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for debugging
            continue

    print("Patch-based inference completed.")


def calculate_metrics(gt_img, pred_img):
    """Calculates PSNR and SSIM between two images using the Y channel."""
    if gt_img.size != pred_img.size:
        pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

    # Convert images to YCbCr
    gt_ycbcr = gt_img.convert("YCbCr")
    pred_ycbcr = pred_img.convert("YCbCr")

    # Extract Y channel
    gt_y = np.array(gt_ycbcr)[:, :, 0].astype(np.float32)
    pred_y = np.array(pred_ycbcr)[:, :, 0].astype(np.float32)

    # Calculate PSNR and SSIM on the Y channel
    psnr_value = psnr(gt_y, pred_y, data_range=255)
    ssim_value = ssim(
        gt_y, pred_y, data_range=255, gaussian_weights=True, use_sample_covariance=False
    )

    return psnr_value, ssim_value


def process_image(img_name, gt_path, pred_path):
    """Loads images, calculates metrics, and returns results."""
    gt_img_path = os.path.join(gt_path, img_name)
    pred_img_path = os.path.join(pred_path, img_name)

    if not os.path.exists(pred_img_path):
        return img_name, None, None, None, None

    try:
        # 优化图像加载
        gt_img = Image.open(gt_img_path).convert("RGB")
        pred_img = Image.open(pred_img_path).convert("RGB")
        psnr_value, ssim_value = calculate_metrics(gt_img, pred_img)
        return img_name, psnr_value, ssim_value, gt_img, pred_img
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        return img_name, None, None, None, None


def calculate_all_metrics(gt_path, pred_path):
    """Calculates metrics for all image pairs in the given directories."""
    # 预过滤文件
    gt_files = [f for f in os.listdir(gt_path) if f.endswith((".png", ".jpg"))]
    pred_files = [f for f in os.listdir(pred_path) if f.endswith((".png", ".jpg"))]
    # 只处理两个目录中都存在的文件
    image_files = [f for f in gt_files if f in pred_files]
    results = []

    # 使用ProcessPoolExecutor而不是ThreadPoolExecutor处理CPU密集型任务
    # 根据CPU核心数和图像数量优化线程数
    max_workers = min(os.cpu_count(), 16)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_image, img_name, gt_path, pred_path)
            for img_name in image_files
        ]

        # 使用as_completed而不是wait，可以更早地开始处理结果
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(image_files),
            desc="Calculating Metrics",
        ):
            results.append(future.result())

    return results


def display_images(gt_img, pred_img, psnr_val, ssim_val, ax):
    """Displays images and metrics on a single axis."""
    combined_img = np.vstack((np.array(gt_img), np.array(pred_img)))
    ax.imshow(combined_img)
    ax.axis("off")

    text_y = gt_img.size[1]
    ax.text(
        gt_img.size[0] / 2,
        text_y + 10,
        f"PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.2f}",
        fontsize=10,
        ha="center",
        va="top",
        bbox=dict(
            facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"
        ),
    )
    ax.text(
        gt_img.size[0] / 2,
        0,
        "Ground Truth",
        fontsize=12,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )
    ax.text(
        gt_img.size[0] / 2,
        text_y * 2,
        "Prediction",
        fontsize=12,
        ha="center",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )


def display_top_bottom(values, metric_name, num_cols, reverse_sort=False):
    """Displays top/bottom images based on the specified metric."""
    if not values:
        print(f"No data to display for {metric_name}.")
        return

    print(f"\n--- Displaying Top/Bottom {num_cols} {metric_name} Images ---")
    num_rows = (min(num_cols, len(values)) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 5 * num_rows))

    if (
        num_rows * num_cols > 1
    ):  # Avoid error when num_rows * num_cols == 1, axes will not be a array
        axes = axes.flatten()
    elif num_rows * num_cols == 1:  # when is 1, make axes become an array.
        axes = [axes]

    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    metric_index = 1 if metric_name == "PSNR" else 2

    # 优化排序：先过滤掉None值，再一次性排序
    filtered_values = [v for v in values if v[metric_index] is not None]
    filtered_values.sort(key=lambda x: x[metric_index], reverse=reverse_sort)

    valid_count = 0
    for i in range(len(filtered_values)):
        if valid_count >= num_cols:
            break  # Stop once we've displayed enough

        img_data = filtered_values[i]
        if img_data[3] is not None and img_data[4] is not None:
            display_images(
                img_data[3], img_data[4], img_data[1], img_data[2], axes[valid_count]
            )
            valid_count += 1

    for j in range(valid_count, len(axes)):
        axes[j].axis("off")

    plt.show()


def plot_histograms(psnr_data, ssim_data):
    """Plots histograms of PSNR and SSIM distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(psnr_data, kde=True, ax=axes[0])
    axes[0].set_title("PSNR Distribution")
    axes[0].set_xlabel("PSNR")
    axes[0].set_ylabel("Frequency")

    sns.histplot(ssim_data, kde=True, ax=axes[1])
    axes[1].set_title("SSIM Distribution")
    axes[1].set_xlabel("SSIM")
    axes[1].set_ylabel("Frequency")
    plt.show()


def analyze_data(gt_path, pred_path):
    """Main function to analyze data and optionally display results."""
    results = calculate_all_metrics(gt_path, pred_path)

    # 过滤并提前计算统计值
    valid_results = [r for r in results if r[1] is not None]
    if not valid_results:
        print("No matching images found for evaluation.")
        return

    # 转换为numpy数组以提高统计计算效率
    psnr_values = np.array([x[1] for x in valid_results])
    ssim_values = np.array([x[2] for x in valid_results])

    # 统计计算
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    max_psnr = np.max(psnr_values)
    min_psnr = np.min(psnr_values)
    max_ssim = np.max(ssim_values)
    min_ssim = np.min(ssim_values)

    # 输出结果
    print(f"\nAverage PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("--- PSNR Statistics ---")
    print(f"  Max:     {max_psnr:.4f}")
    print(f"  Min:     {min_psnr:.4f}")
    print("--- SSIM Statistics ---")
    print(f"  Max:     {max_ssim:.4f}")
    print(f"  Min:     {min_ssim:.4f}")

    # --- Best and Worst Cases ---
    print("\n--- Top 4 Best and Worst Cases ---")

    # 使用数组索引优化排序
    psnr_indices = np.argsort(psnr_values)
    top3_psnr = [valid_results[i] for i in psnr_indices[-4:]][::-1]
    worst3_psnr = [valid_results[i] for i in psnr_indices[:4]]

    ssim_indices = np.argsort(ssim_values)
    top3_ssim = [valid_results[i] for i in ssim_indices[-4:]][::-1]
    worst3_ssim = [valid_results[i] for i in ssim_indices[:4]]

    print("\nTop 3 PSNR:")
    for i, res in enumerate(top3_psnr):
        print(f"  {i + 1}: {res[0]} ({res[1]:.4f})")

    print("\nWorst 3 PSNR:")
    for i, res in enumerate(worst3_psnr):
        print(f"  {i + 1}: {res[0]} ({res[1]:.4f})")

    print("\nTop 3 SSIM:")
    for i, res in enumerate(top3_ssim):
        print(f"  {i + 1}: {res[0]} ({res[2]:.4f})")

    print("\nWorst 3 SSIM:")
    for i, res in enumerate(worst3_ssim):
        print(f"  {i + 1}: {res[0]} ({res[2]:.4f})")

    # 转回列表保持一致性
    psnr_values = psnr_values.tolist()
    ssim_values = ssim_values.tolist()
    return valid_results, psnr_values, ssim_values
