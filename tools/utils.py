import glob
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import concurrent.futures  # For multithreading
import cv2  # Use OpenCV for faster saving
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model, model_path):
    ckpt = torch.load(model_path, weights_only=False, map_location="cpu")["state_dict"]
    for k in list(ckpt.keys()):
        if "model" not in k:
            ckpt.pop(k)
        else:
            ckpt[k.replace("model.", "")] = ckpt.pop(k)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def save_image_cv2(img_array, output_path):
    """Saves an image using OpenCV."""
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    cv2.imwrite(output_path, img_array)


def infer_from_model(
    model,
    images_path,
    output_path,
    tta=False,
    scale_factor=4,
    device="cuda:0",
    num_workers=4,  # Add num_workers for multithreading
):
    # 设备选择
    model = model.to(device)

    # 获取图像列表
    if os.path.isdir(images_path):
        image_files = sorted(
            glob.glob(os.path.join(images_path, "*.png"))
            + glob.glob(os.path.join(images_path, "*.jpg"))
        )
    else:
        image_files = [images_path]

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for img_path in tqdm(image_files, desc="Inferencing"):
            # 加载图像
            img = Image.open(img_path).convert("RGB")
            img_tensor = transforms.ToTensor()(img).to(device) * 2 - 1.0

            # 添加batch维度
            img_tensor = img_tensor.unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)

            # 后处理
            output = (output[0] + 1.0) * 127.5  # [-1,1] -> [0,255]
            output = output.clamp(0, 255).cpu().numpy()
            output = np.transpose(output, (1, 2, 0)).astype(np.uint8)

            # --- Use OpenCV and Multithreading for Saving ---
            img_name = img_path.split("/")[-1]
            output_file_path = os.path.join(output_path, img_name)
            executor.submit(save_image_cv2, output, output_file_path)
        executor.shutdown(wait=True)


def calculate_metrics(gt_img, pred_img):
    """Calculates PSNR and SSIM between two images."""
    if gt_img.size != pred_img.size:
        pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

    gt_np = np.array(gt_img).astype(np.float32)
    pred_np = np.array(pred_img).astype(np.float32)

    psnr_value = psnr(gt_np, pred_np, data_range=255)

    if len(gt_np.shape) == 3:  # Color image
        ssim_value = 0
        for i in range(gt_np.shape[2]):
            ssim_value += ssim(gt_np[:, :, i], pred_np[:, :, i], data_range=255)
        ssim_value /= gt_np.shape[2]
    else:  # Grayscale image
        ssim_value = ssim(gt_np, pred_np, data_range=255)

    return psnr_value, ssim_value


def process_image(img_name, gt_path, pred_path):
    """Loads images, calculates metrics, and returns results."""
    gt_img_path = os.path.join(gt_path, img_name)
    pred_img_path = os.path.join(pred_path, img_name.replace(".png", "x4.png"))

    if os.path.exists(pred_img_path):
        gt_img = Image.open(gt_img_path).convert("RGB")
        pred_img = Image.open(pred_img_path).convert("RGB")
        psnr_value, ssim_value = calculate_metrics(gt_img, pred_img)
        return img_name, psnr_value, ssim_value, gt_img, pred_img
    return img_name, None, None, None, None


def calculate_all_metrics(gt_path, pred_path):
    """Calculates metrics for all image pairs in the given directories."""
    image_files = [f for f in os.listdir(gt_path) if f.endswith((".png", ".jpg"))]
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(process_image, img_name, gt_path, pred_path)
            for img_name in image_files
        ]
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
    values.sort(
        key=lambda x: x[metric_index]
        if x[metric_index] is not None
        else -np.inf
        if reverse_sort
        else np.inf,
        reverse=reverse_sort,
    )  # Handle None values

    valid_count = 0
    for i in range(len(values)):
        if valid_count >= num_cols:
            break  # Stop once we've displayed enough

        img_data = values[i]
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

    # Filter out None results and prepare data for analysis
    valid_results = [r for r in results if r[1] is not None]
    if not valid_results:
        print("No matching images found for evaluation.")
        return

    psnr_values = [x[1] for x in valid_results]
    ssim_values = [x[2] for x in valid_results]

    # --- Calculate and print statistics ---
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    # print(f"{avg_psnr:.4f}\n")
    # print(f"{avg_ssim:.4f}\n")
    # print(f"{np.max(psnr_values):.4f}\n")
    # print(f"{np.min(psnr_values):.4f}\n")
    # print(f"{np.max(ssim_values):.4f}\n")
    # print(f"{np.min(ssim_values):.4f}\n")
    print(f"\nAverage PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("--- PSNR Statistics ---")
    print(f"  Max:     {np.max(psnr_values):.4f}")
    print(f"  Min:     {np.min(psnr_values):.4f}")
    print("--- SSIM Statistics ---")
    print(f"  Max:     {np.max(ssim_values):.4f}")
    print(f"  Min:     {np.min(ssim_values):.4f}")

    # --- Best and Worst Cases ---
    print("\n--- Top 4 Best and Worst Cases ---")
    top3_psnr = sorted(valid_results, key=lambda x: x[1], reverse=True)[:4]
    worst3_psnr = sorted(valid_results, key=lambda x: x[1])[:4]
    top3_ssim = sorted(valid_results, key=lambda x: x[2], reverse=True)[:4]
    worst3_ssim = sorted(valid_results, key=lambda x: x[2])[:4]

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
    return valid_results, psnr_values, ssim_values
