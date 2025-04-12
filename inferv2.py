import os
import argparse
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.gridspec import GridSpec
import concurrent.futures
import base64
from io import BytesIO
import torch
import cv2


# ===============================
# 1. 工具函数模块
# ===============================
class ImageUtils:
    """通用图像处理工具类，提供各个模块共享的图像处理功能"""

    @staticmethod
    def load_image(path, convert_to_rgb=False, as_bgr=False):
        """加载图像，支持错误处理和可选的颜色转换

        Args:
            path: 图像文件路径
            convert_to_rgb: 是否转换为RGB（PIL）
            as_bgr: 是否以BGR格式返回（OpenCV）

        Returns:
            如果as_bgr=True，返回OpenCV格式的图像（numpy数组），否则返回PIL图像
        """
        try:
            if as_bgr:  # 使用OpenCV加载为BGR
                img = cv2.imread(path)
                if img is None:
                    raise IOError(f"Cannot read image: {path}")
                return img
            else:  # 使用PIL加载
                img = Image.open(path)
                if convert_to_rgb:
                    img = img.convert("RGB")
                return img
        except Exception as e:
            print(f"Failed to load image '{path}': {e}")
            return None

    @staticmethod
    def img_to_base64(img):
        """将PIL图像转换为base64字符串，用于在HTML中嵌入"""
        try:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            print(f"Failed to convert image to base64: {e}")
            return ""

    @staticmethod
    def plt_figure_to_base64(fig):
        """将matplotlib图形转换为base64字符串"""
        try:
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.close(fig)  # 关闭图形以释放内存
            return img_str
        except Exception as e:
            print(f"Failed to convert figure to base64: {e}")
            plt.close(fig)  # 确保即使出错也关闭图形
            return ""

    @staticmethod
    def extract_patch(img, patch_size=(256, 256), method="center"):
        """从图像中提取指定大小的区域

        Args:
            img: PIL图像
            patch_size: (宽度, 高度)元组
            method: 提取方法 - 目前支持"center"

        Returns:
            提取的PIL图像块，大小精确为patch_size
        """
        if not isinstance(img, Image.Image):
            raise TypeError("Input image must be a PIL Image")

        width, height = img.size
        target_w, target_h = patch_size

        if method == "center":
            # 计算中心区域的坐标
            left = (width - target_w) // 2
            top = (height - target_h) // 2
            right = left + target_w
            bottom = top + target_h

            # 处理图像小于目标大小的情况
            if width < target_w or height < target_h:
                # 如果图像较小，先按比例放大，然后裁剪中心区域
                scale = max(target_w / width, target_h / height)
                new_w = int(width * scale)
                new_h = int(height * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)  # 使用高质量的调整大小
                width, height = img.size  # 更新尺寸
                # 重新计算放大后图像的裁剪框
                left = (width - target_w) // 2
                top = (height - target_h) // 2
                right = left + target_w
                bottom = top + target_h

            # 确保裁剪框在图像范围内
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)

            cropped = img.crop((left, top, right, bottom))

            # 确保最终输出精确为patch_size，如果裁剪较小则调整大小
            if cropped.size != patch_size:
                cropped = cropped.resize(patch_size, Image.BICUBIC)

            return cropped
        else:
            raise ValueError(f"Unsupported patch extraction method: {method}")

    @staticmethod
    def upscale_with_interpolation(lr_img, hr_size, method="bicubic"):
        """使用指定的插值方法放大低分辨率图像

        Args:
            lr_img: 低分辨率PIL图像
            hr_size: 目标大小(宽度, 高度)
            method: 插值方法（nearest, bilinear, bicubic, lanczos）

        Returns:
            插值放大后的PIL图像
        """
        interpolation_methods = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        interp_method = interpolation_methods.get(method.lower(), Image.BICUBIC)
        return lr_img.resize(hr_size, interp_method)

    @staticmethod
    def convert_input_type_range(img):
        """转换图像的类型和范围

        将输入图像转换为np.float32类型和[0, 1]范围。主要用于预处理色彩空间转换
        函数（如rgb2ycbcr和ycbcr2rgb）中的输入图像。

        Args:
            img (ndarray): 输入图像。接受：
                1. np.uint8类型，范围[0, 255]；
                2. np.float32类型，范围[0, 1]。

        Returns:
            (ndarray): 转换后的图像，类型为np.float32，范围为[0, 1]。
        """
        img_type = img.dtype
        img = img.astype(np.float32)
        if img_type == np.float32:
            pass
        elif img_type == np.uint8:
            img /= 255.0
        else:
            raise TypeError(
                f"The img type should be np.float32 or np.uint8, but got {img_type}"
            )
        return img

    @staticmethod
    def convert_output_type_range(img, dst_type):
        """根据dst_type转换图像的类型和范围

        它将图像转换为所需的类型和范围。如果`dst_type`是np.uint8，
        图像将被转换为np.uint8类型，范围[0, 255]。如果`dst_type`
        是np.float32，它将图像转换为np.float32类型，范围[0, 1]。

        Args:
            img (ndarray): 要转换的图像，np.float32类型，范围[0, 255]。
            dst_type (np.uint8 | np.float32): 如果dst_type是np.uint8，它
                将图像转换为np.uint8类型，范围[0, 255]。如果dst_type是
                np.float32，它将图像转换为np.float32类型，范围[0, 1]。

        Returns:
            (ndarray): 转换后的图像，具有所需的类型和范围。
        """
        if dst_type not in (np.uint8, np.float32):
            raise TypeError(
                f"The dst_type should be np.float32 or np.uint8, but got {dst_type}"
            )
        if dst_type == np.uint8:
            img = img.round()
        else:
            img /= 255.0
        return img.astype(dst_type)

    @staticmethod
    def bgr2ycbcr(img, y_only=False):
        """将BGR图像转换为YCbCr图像

        实现ITU-R BT.601标准定义电视转换。详见
        https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion

        与cv2.cvtColor中的类似函数不同：`BGR <-> YCrCb`。
        在OpenCV中，它实现了JPEG转换。详见
        https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion

        Args:
            img (ndarray): 输入图像。接受：
                1. np.uint8类型，范围[0, 255]；
                2. np.float32类型，范围[0, 1]。
            y_only (bool): 是否只返回Y通道。默认：False。

        Returns:
            ndarray: 转换后的YCbCr图像。输出图像具有与输入图像相同的类型和范围。
        """
        img_type = img.dtype
        img = ImageUtils.convert_input_type_range(img)
        if y_only:
            out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
        else:
            out_img = np.matmul(
                img,
                [
                    [24.966, 112.0, -18.214],
                    [128.553, -74.203, -93.786],
                    [65.481, -37.797, 112.0],
                ],
            ) + [16, 128, 128]
        out_img = ImageUtils.convert_output_type_range(out_img, img_type)

        return out_img

    @staticmethod
    def to_y_channel(img):
        """转换为YCbCr的Y通道

        Args:
            img (ndarray): 范围为[0, 255]的图像。

        Returns:
            (ndarray): 范围为[0, 255]的浮点类型图像，无四舍五入。
        """
        img = img.astype(np.float32) / 255.0
        if img.ndim == 3 and img.shape[2] == 3:
            img = ImageUtils.bgr2ycbcr(img, y_only=True)
        return img * 255.0


# ===============================
# 2. 指标计算模块
# ===============================
class MetricsCalculator:
    """图像质量评估指标计算类，支持PSNR和SSIM等指标"""

    @staticmethod
    def _ssim(img, img2):
        """计算单通道图像的SSIM（结构相似性）

        由函数`calculate_ssim`调用。

        Args:
            img (ndarray): 范围为[0, 255]的'HWC'顺序图像。
            img2 (ndarray): 范围为[0, 255]的'HWC'顺序图像。

        Returns:
            float: ssim结果。
        """
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        img = img.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return ssim_map.mean()

    @staticmethod
    def calculate_metrics(
        gt_img, pred_img, metrics=None, y_channel=True, crop_border=6
    ):
        """在两个图像间计算质量评估指标

        Args:
            gt_img: 参考图像（真实图像），PIL Image或numpy数组
            pred_img: 测试图像（预测图像），PIL Image或numpy数组
            metrics: 要计算的指标列表，默认["psnr", "ssim"]
            y_channel: 是否只在Y通道上计算指标

        Returns:
            返回计算的指标字典，键为指标名称，值为指标值
        """
        if metrics is None:
            metrics = ["psnr", "ssim"]

        # 标准化输入，确保是OpenCV格式的numpy数组（BGR顺序）
        if isinstance(gt_img, Image.Image):
            gt_img = np.array(gt_img)[:, :, ::-1]  # RGB to BGR
        if isinstance(pred_img, Image.Image):
            pred_img = np.array(pred_img)[:, :, ::-1]  # RGB to BGR

        results = {}

        # 转换为Y通道进行评估（如有需要）
        if y_channel:
            gt_y = ImageUtils.to_y_channel(gt_img)[
                crop_border:-crop_border, crop_border:-crop_border
            ]  # 移除边界
            pred_y = ImageUtils.to_y_channel(pred_img)[
                crop_border:-crop_border, crop_border:-crop_border
            ]  # 移除边界

            eval_gt = gt_y
            eval_pred = pred_y
        else:
            # 使用完整RGB图像
            eval_gt = gt_img
            eval_pred = pred_img

        # 计算PSNR
        if "psnr" in metrics:
            try:
                mse = np.mean((eval_gt - eval_pred) ** 2)
                if mse == 0:
                    psnr_value = float("inf")
                else:
                    psnr_value = 10.0 * np.log10(255.0 * 255.0 / mse)
                results["psnr"] = psnr_value
            except Exception as e:
                print(f"PSNR计算错误: {e}")
                results["psnr"] = 0.0

        # 计算SSIM
        if "ssim" in metrics:
            try:
                ssim_value = MetricsCalculator._ssim(eval_gt, eval_pred)
                results["ssim"] = ssim_value
            except Exception as e:
                print(f"SSIM计算错误: {e}")
                results["ssim"] = 0.0

        return results

    @staticmethod
    def calculate_metrics_y_channel(gt_img, pred_img):
        """计算Y通道图像质量指标的简便方法（兼容旧代码）

        Args:
            gt_img: 参考图像
            pred_img: 预测图像

        Returns:
            (psnr_value, ssim_value) 元组
        """
        metrics = MetricsCalculator.calculate_metrics(
            gt_img, pred_img, y_channel=True, crop_border=9
        )
        return metrics.get("psnr", 0.0), metrics.get("ssim", 0.0)

    @staticmethod
    def process_single_image(args):
        """处理单张图像的指标计算，用于并行处理

        Args:
            args: (img_path, hr_dir, pred_dir) 元组

        Returns:
            (img_name, psnr_value, ssim_value, base_name) 结果元组
        """
        img_path, hr_dir, pred_dir = args
        img_name = os.path.basename(img_path)
        hr_path = os.path.join(hr_dir, img_name)
        try:
            hr_img = ImageUtils.load_image(hr_path, as_bgr=True)
            pred_img = ImageUtils.load_image(img_path, as_bgr=True)

            if hr_img is None or pred_img is None:
                raise FileNotFoundError(f"无法加载图像 {img_name}")

            psnr_value, ssim_value = MetricsCalculator.calculate_metrics_y_channel(
                hr_img, pred_img
            )
            return img_name, psnr_value, ssim_value, img_name.split(".")[0]
        except FileNotFoundError:
            print(f"错误：HR图像未找到：{img_name}，路径：{hr_path}")
            return img_name, None, None, img_name.split(".")[0]
        except Exception as e:
            print(f"处理 {img_path} 出错: {e}")
            return img_name, None, None, img_name.split(".")[0]

    @staticmethod
    def evaluate_sr_results(dataset_dir, pred_dir, scale, num_workers=None):
        """评估超分辨率结果

        Args:
            dataset_dir: 数据集根目录（包含HR子目录）
            pred_dir: 预测结果目录
            scale: 超分辨率缩放因子
            num_workers: 并行处理的工作线程数

        Returns:
            评估结果列表，每项为(img_name, psnr, ssim, base_name)元组
        """
        hr_dir = os.path.join(dataset_dir, "HR")

        if not os.path.exists(hr_dir):
            print(f"错误: HR 目录不存在: {hr_dir}")
            return None
        if not os.path.exists(pred_dir):
            print(f"错误: 预测目录不存在: {pred_dir}")
            return None

        pred_files = (
            glob.glob(os.path.join(pred_dir, "*.png"))
            + glob.glob(os.path.join(pred_dir, "*.jpg"))
            + glob.glob(os.path.join(pred_dir, "*.bmp"))
        )
        if not pred_files:
            print(f"在 {pred_dir} 中未找到预测图像 (png, jpg, bmp)")
            return None

        results = []
        args_list = [(pred_path, hr_dir, pred_dir) for pred_path in pred_files]

        if num_workers is None:
            num_workers = os.cpu_count()
        # 限制工作线程数以避免资源耗尽，特别是内存
        num_workers = min(num_workers, 8)  # 根据需要调整此限制

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 使用列表确保所有future在继续之前完成
            futures = [
                executor.submit(MetricsCalculator.process_single_image, arg)
                for arg in args_list
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(args_list),
                desc=f"评估 {os.path.basename(dataset_dir)} x{scale}",
            ):
                result = future.result()
                if (
                    result[1] is not None and result[2] is not None
                ):  # 检查PSNR和SSIM是否都有效
                    results.append(result)
                else:
                    print(f"由于处理错误，跳过 {result[0]} 的结果。")

        return results


# ===============================
# 3. 可视化模块
# ===============================
class Visualizer:
    """数据可视化类，用于创建图表和比较展示"""

    @staticmethod
    def plot_dataset_results(
        dataset_results,
        dataset_name,
        scale,
        figsize=(12, 7),
        dpi=120,
    ):
        """Plot the distribution of dataset evaluation results

        Args:
            dataset_results: List of evaluation results
            dataset_name: Name of the dataset
            scale: Scaling factor
            figsize: Figure size tuple
            dpi: Figure DPI

        Returns:
            matplotlib Figure object
        """
        if not dataset_results or len(dataset_results) == 0:
            print(f"Cannot plot for {dataset_name}: No valid results")
            return None

        psnr_values = [r[1] for r in dataset_results if r[1] is not None]
        ssim_values = [r[2] for r in dataset_results if r[2] is not None]

        if not psnr_values or not ssim_values:
            print(f"Cannot plot for {dataset_name}: No valid PSNR or SSIM values")
            return None

        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(
            2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1]
        )  # Adjust ratios as needed

        # Configure fonts (optional, but helps maintain consistency)
        plt.rcParams.update(
            {"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10}
        )

        # --- PSNR Distribution ---
        ax1 = fig.add_subplot(gs[0, 0])
        mean_psnr = np.mean(psnr_values)
        sns.histplot(
            psnr_values, kde=True, ax=ax1, bins=20, color="skyblue", edgecolor="black"
        )  # Better histogram/colors
        ax1.axvline(
            mean_psnr,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_psnr:.2f} dB",
        )
        ax1.set_title(f"{dataset_name} PSNR Distribution (x{scale})", fontweight="bold")
        ax1.set_xlabel("PSNR (dB)")
        ax1.set_ylabel("Frequency")
        ax1.legend(fontsize=9)
        ax1.grid(axis="y", linestyle="--", alpha=0.6)

        # --- SSIM Distribution ---
        ax2 = fig.add_subplot(gs[0, 1])
        mean_ssim = np.mean(ssim_values)
        sns.histplot(
            ssim_values,
            kde=True,
            ax=ax2,
            bins=20,
            color="lightcoral",
            edgecolor="black",
        )
        ax2.axvline(
            mean_ssim,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_ssim:.4f}",
        )
        ax2.set_title(f"{dataset_name} SSIM Distribution (x{scale})", fontweight="bold")
        ax2.set_xlabel("SSIM")
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", linestyle="--", alpha=0.6)

        # --- PSNR vs SSIM Scatter Plot ---
        ax3 = fig.add_subplot(gs[1, :])
        scatter = ax3.scatter(
            psnr_values,
            ssim_values,
            alpha=0.7,
            c=psnr_values,
            cmap="viridis",
            s=35,
            edgecolors="grey",
            linewidth=0.5,  # Add border color
        )
        ax3.set_title(f"{dataset_name} PSNR vs SSIM (x{scale})", fontweight="bold")
        ax3.set_xlabel("PSNR (dB)")
        ax3.set_ylabel("SSIM")
        ax3.grid(True, linestyle="--", alpha=0.6)
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3, label="PSNR (dB)")
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label("PSNR (dB)", size=10)

        plt.tight_layout(pad=2.0)  # Add padding
        return fig

    @staticmethod
    def generate_cases_analysis_html(
        result_dir,
        dataset_dir,
        results,
        scale,
        case_type="best",
        patch_size=(256, 256),
        interp_method="bicubic",
        num_cases=12,
    ):
        """生成带有放大镜功能的最佳/最差案例的HTML

        Args:
            result_dir: 超分辨率结果目录
            dataset_dir: 数据集目录
            results: 评估结果列表
            scale: 放大比例
            case_type: 案例类型，"best"或"worst"
            patch_size: 显示的图像块大小
            interp_method: 用于比较的插值方法
            num_cases: 显示的案例数量

        Returns:
            HTML内容字符串
        """
        if not results or len(results) == 0:
            return f"<div class='error'>没有可用于{case_type}案例分析的结果。</div>"

        # 按PSNR排序结果
        sorted_results = sorted(
            results,
            key=lambda x: x[1] if x[1] is not None else -float("inf"),
            reverse=(case_type == "best"),
        )

        # 选择案例
        selected_cases = sorted_results[: min(num_cases, len(sorted_results))]
        if len(selected_cases) < 1:
            return f"<div class='info'>没有足够的有效结果({len(results)})来显示{case_type}案例。</div>"

        interp_method_display = interp_method.capitalize()
        patch_w, patch_h = patch_size

        if case_type == "best":
            title = f"表现最佳的 {len(selected_cases)} 个案例（按完整图像PSNR排序）"
        else:
            title = f"表现最差的 {len(selected_cases)} 个案例（按完整图像PSNR排序）"

        html_content = f"""
        <div class="case-analysis-section">
            <h4>{title}</h4>
            <p class="case-analysis-info">显示中心 {patch_w}x{patch_h} 区域。比较使用 {interp_method_display} 插值。 
            使用页面顶部的放大镜控件检查细节。</p>
            <div class="cases-grid">
        """

        # 处理选定的案例
        processed_count = 0
        for rank, (img_name, full_psnr, full_ssim, _) in enumerate(selected_cases):
            if full_psnr is None or full_ssim is None:
                continue  # 如果指标缺失则跳过

            try:
                hr_path = os.path.join(dataset_dir, "HR", img_name)
                sr_path = os.path.join(result_dir, f"PRED_x{scale}", img_name)
                lr_path = os.path.join(dataset_dir, f"x{scale}", img_name)

                # 检查文件是否存在
                if not os.path.exists(hr_path):
                    raise FileNotFoundError(f"未找到HR: {hr_path}")
                if not os.path.exists(sr_path):
                    raise FileNotFoundError(f"未找到SR: {sr_path}")
                if not os.path.exists(lr_path):
                    raise FileNotFoundError(f"未找到LR: {lr_path}")

                hr_img = ImageUtils.load_image(hr_path, convert_to_rgb=True)
                sr_img = ImageUtils.load_image(sr_path, convert_to_rgb=True)
                lr_img = ImageUtils.load_image(lr_path, convert_to_rgb=True)

                hr_size = hr_img.size
                # 使用指定的插值方法放大LR图像
                lr_upscaled = ImageUtils.upscale_with_interpolation(
                    lr_img, hr_size, interp_method
                )

                # 提取中心区域
                hr_center = ImageUtils.extract_patch(hr_img, patch_size)
                sr_center = ImageUtils.extract_patch(sr_img, patch_size)
                interp_center = ImageUtils.extract_patch(lr_upscaled, patch_size)

                # 计算中心区域的指标
                patch_metrics_sr = MetricsCalculator.calculate_metrics(
                    hr_center, sr_center
                )
                patch_metrics_interp = MetricsCalculator.calculate_metrics(
                    hr_center, interp_center
                )

                patch_psnr_sr = patch_metrics_sr.get("psnr", 0)
                patch_ssim_sr = patch_metrics_sr.get("ssim", 0)
                patch_psnr_interp = patch_metrics_interp.get("psnr", 0)
                patch_ssim_interp = patch_metrics_interp.get("ssim", 0)

                # 准备base64字符串
                hr_base64 = ImageUtils.img_to_base64(hr_center)
                sr_base64 = ImageUtils.img_to_base64(sr_center)
                interp_base64 = ImageUtils.img_to_base64(interp_center)

                # 计算改进
                psnr_improvement = patch_psnr_sr - patch_psnr_interp
                ssim_improvement = patch_ssim_sr - patch_ssim_interp

                # 确定改进指示的类
                psnr_class = (
                    "improvement positive"
                    if psnr_improvement > 0
                    else "improvement negative"
                )
                ssim_class = (
                    "improvement positive"
                    if ssim_improvement > 0
                    else "improvement negative"
                )

                html_content += f"""
                <div class="case-item">
                    <div class="case-header">
                        <h5>{rank + 1}. {img_name}</h5>
                        <div class="full-metrics">
                            完整图像: PSNR: {full_psnr:.2f} dB / SSIM: {full_ssim:.4f}
                        </div>
                    </div>
                    
                    <div class="case-comparison" data-case-id="{img_name.replace(".", "_")}">
                        <div class="image-col">
                            <img src="data:image/png;base64,{interp_base64}" alt="{interp_method_display} Patch">
                            <div class="image-label">{interp_method_display}</div>
                            <div class="patch-metrics">
                                PSNR: {patch_psnr_interp:.2f} dB<br>SSIM: {patch_ssim_interp:.4f}
                            </div>
                        </div>
                        
                        <div class="image-col">
                            <img src="data:image/png;base64,{sr_base64}" alt="SR Patch">
                            <div class="image-label">SR (本模型)</div>
                            <div class="patch-metrics">
                                <span>PSNR: {patch_psnr_sr:.2f} dB</span> 
                                <small class="{psnr_class}">({"+" if psnr_improvement > 0 else ""}{psnr_improvement:.2f})</small><br>
                                <span>SSIM: {patch_ssim_sr:.4f}</span>
                                <small class="{ssim_class}">({"+" if ssim_improvement > 0 else ""}{ssim_improvement:.4f})</small>
                            </div>
                        </div>
                        
                        <div class="image-col">
                            <img src="data:image/png;base64,{hr_base64}" alt="GT Patch">
                            <div class="image-label">Ground Truth (GT)</div>
                            <div class="patch-metrics">
                                (参考)
                            </div>
                        </div>
                    </div>
                </div>
                """
                processed_count += 1

            except FileNotFoundError as fnf_err:
                html_content += (
                    f"<div class='case-item error'>跳过 {img_name}: {fnf_err}</div>"
                )
            except Exception as e:
                html_content += (
                    f"<div class='case-item error'>处理 {img_name} 时出错: {e}</div>"
                )

        if processed_count == 0 and len(selected_cases) > 0:
            html_content += f"<div class='error'>由于错误（例如缺少文件），无法处理所选的 {len(selected_cases)} 个案例中的任何一个。</div>"

        html_content += """
            </div>
        </div>
        """
        return html_content


# ===============================
# 4. HTML报告生成模块
# ===============================
class ReportGenerator:
    """HTML报告生成类，用于创建交互式评估报告"""

    # HTML样式和JavaScript模板（由于太长，这里省略，在create_html_report中使用）

    @staticmethod
    def create_html_report(
        all_results,
        output_path,
        dataset_path="SR_Test",
        model_name="Model",
        patch_size=(256, 256),
        interp_method="bicubic",
        num_cases=12,
    ):
        """生成改进的HTML评估报告

        Args:
            all_results: 评估结果字典
            output_path: 输出目录
            dataset_path: 数据集路径
            model_name: 模型名称
            patch_size: 案例分析中的图像块大小
            interp_method: 用于比较的插值方法
            num_cases: 每个数据集显示的最佳/最差案例数量
        """

        # --- 增强的CSS ---
        style = """
        <style>
        :root {
            /* Modern color scheme */
            --primary-color: #3498db; /* Blue */
            --primary-dark: #2980b9;
            --secondary-color: #2ecc71; /* Green */
            --secondary-dark: #27ae60;
            --accent-color: #e74c3c;  /* Red for attention */
            --light-gray: #f5f7fa;
            --medium-gray: #e9ecef;
            --dark-gray: #6c757d;
            --text-color: #2d3436;
            --white: #ffffff;
            --border-color: #dee2e6;
            --success-light: #d4edda;
            --danger-light: #f8d7da;
            --danger-dark: #721c24;
            --font-sans-serif: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-chinese: "Microsoft YaHei", "SimHei", "PingFang SC", sans-serif;
            
            /* Shadows */
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08);
            --shadow-md: 0 4px 6px rgba(0,0,0,0.1), 0 1px 3px rgba(0,0,0,0.08);
            --shadow-lg: 0 10px 15px rgba(0,0,0,0.07), 0 4px 6px rgba(0,0,0,0.05);
        }

        body {
            font-family: var(--font-sans-serif);
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--light-gray);
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 1400px;
            margin: 20px auto;
            padding: 1.5rem;
            background-color: #fff;
            border-radius: 12px;
            box-shadow: var(--shadow-lg);
        }
        @media (min-width: 1600px) { .container { max-width: 1500px; } }
        @media (min-width: 1900px) { .container { max-width: 1700px; } }

        h1, h2, h3, h4, h5, h6 {
            font-family: var(--font-chinese), var(--font-sans-serif);
            color: var(--text-color);
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            font-weight: 600;
            line-height: 1.3;
        }
        h1 { font-size: 2.2rem; color: var(--primary-color); border-bottom: 2px solid var(--primary-color); padding-bottom: 0.5rem;}
        h2 { font-size: 1.8rem; color: var(--secondary-color); margin-top: 2em; }
        h3 { font-size: 1.5rem; }
        h4 { font-size: 1.3rem; margin-top: 1em; margin-bottom: 0.5em;}
        h5 { font-size: 1.1rem; margin-bottom: 0.3em; }

        /* --- Header --- */
        .report-header {
            background: linear-gradient(120deg, var(--primary-color), var(--secondary-color));
            color: #fff;
            padding: 3rem 1.5rem;
            margin-bottom: 2rem;
            border-radius: 12px;
            box-shadow: var(--shadow-md);
            position: relative;
            overflow: hidden;
        }
        .report-header::after {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            bottom: -50%;
            left: -50%;
            background: linear-gradient(to bottom right, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
            transform: rotate(45deg);
            pointer-events: none;
        }
        .report-header h1 { 
            color: #fff; 
            border-bottom: none; 
            margin: 0; 
            padding: 0; 
            font-size: 2.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .report-header p { 
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem; 
            margin-top: 1rem;
            max-width: 800px;
        }

        /* --- Summary Cards --- */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .summary-card {
            background-color: #fff;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--medium-gray);
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .summary-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-md);
        }
        .summary-card h3 {
            margin-top: 0;
            margin-bottom: 1rem;
            color: var(--primary-color);
            font-size: 1.1rem;
            font-weight: 500;
        }
        .summary-card .value {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--secondary-color);
            display: block;
            margin-bottom: 0.25rem;
            line-height: 1.2;
        }
        .summary-card .unit {
            font-size: 0.9rem;
            color: var(--dark-gray);
        }

        /* --- Case Comparison --- */
        .case-item {
            border: 1px solid var(--medium-gray);
            border-radius: 12px;
            background-color: var(--light-gray);
            padding: 1rem;
            transition: all 0.25s ease;
            overflow: hidden;
        }
        .case-item:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-3px);
        }

        .case-comparison {
            display: flex;
            gap: 15px;
            justify-content: space-between;
            position: relative;
        }
        @media (max-width: 500px) { .case-comparison { flex-direction: column; } }

        .image-col {
            flex: 1;
            text-align: center;
            min-width: 0;
            position: relative;
        }
        .image-col img {
            width: 100%;
            height: auto;
            max-height: 250px;
            object-fit: contain;
            border: 1px solid var(--medium-gray);
            border-radius: 8px;
            background-color: #fff;
            margin-bottom: 0.5rem;
            transition: all 0.25s ease;
        }

        /* --- Global Magnifier Control Panel --- */
        .magnifier-control-panel {
            position: sticky;
            top: 0;
            z-index: 1000;
            margin: 0 0 20px 0;
            background-color: white;
            border-radius: 10px;
            box-shadow: var(--shadow-md);
            padding: 1rem;
            transition: all 0.3s ease;
            border: 1px solid var(--border-color);
        }

        .magnifier-controls {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 1.5rem;
            justify-content: center;
        }

        .magnifier-toggle, .magnifier-zoom, .magnifier-size {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .toggle-checkbox {
            appearance: none;
            width: 40px;
            height: 20px;
            background-color: var(--medium-gray);
            border-radius: 20px;
            position: relative;
            cursor: pointer;
            transition: background-color 0.15s;
            margin: 0;
        }

        .toggle-checkbox:checked {
            background-color: var(--secondary-color);
        }

        .toggle-checkbox::before {
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 16px;
            height: 16px;
            background-color: white;
            border-radius: 50%;
            transition: transform 0.15s;
        }

        .toggle-checkbox:checked::before {
            transform: translateX(20px);
        }

        .toggle-label, .zoom-label, .size-label {
            color: var(--text-color);
            font-weight: 500;
            font-size: 0.9rem;
        }

        input[type="range"] {
            width: 120px;
            height: 6px;
            background-color: var(--medium-gray);
            border-radius: 3px;
            outline: none;
            appearance: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 16px;
            height: 16px;
            background-color: var(--primary-color);
            border-radius: 50%;
            cursor: pointer;
        }

        input[type="range"]::-moz-range-thumb {
            width: 16px;
            height: 16px;
            background-color: var(--primary-color);
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }

        #global-zoom-value, #global-size-value {
            min-width: 45px;
            display: inline-block;
            text-align: center;
            font-weight: 600;
            color: var(--primary-color);
        }
        
        /* Magnifier glass */
        .magnifier {
            position: absolute;
            width: 150px;
            height: 150px;
            border-radius: 50%;
            border: 2px solid var(--primary-color);
            background-repeat: no-repeat;
            pointer-events: none;
            box-shadow: 0 0 0 7px rgba(255, 255, 255, 0.85), 0 0 7px 7px rgba(0, 0, 0, 0.25);
            z-index: 9;
            display: none;
        }
        
        /* Improvement indicators */
        .improvement {
            font-size: 0.8rem;
            font-weight: normal;
            padding: 0 4px;
            margin-left: 4px;
            border-radius: 4px;
        }
        .improvement.positive {
            color: var(--secondary-color);
            background-color: rgba(46, 204, 113, 0.1);
        }
        .improvement.negative {
            color: var(--accent-color);
            background-color: rgba(231, 76, 60, 0.1);
        }
            /* --- Tables --- */
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 25px 0;
                background-color: var(--white);
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                border: 1px solid var(--border-color);
                border-radius: 5px; /* Rounded corners for table */
                overflow: hidden; /* Clip content to rounded corners */
            }
            th, td {
                border: 1px solid var(--border-color);
                padding: 12px 15px; /* More padding */
                text-align: center;
                vertical-align: middle;
            }
            th {
                background-color: var(--primary-color);
                color: var(--white);
                font-weight: 600;
                font-size: 0.95rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            tr { background-color: var(--white); }
            tr:nth-child(even) { background-color: var(--light-gray); }
            tr:hover { background-color: var(--medium-gray); }
            td:first-child { text-align: left; }
            table .total-row {
                font-weight: bold;
                background-color: var(--success-light);
                color: #155724; /* Darker green text for contrast */
            }

            /* --- Charts --- */
            .chart-container {
                width: 100%;
                margin: 25px auto; /* Center chart */
                padding: 20px;
                background-color: var(--white);
                border-radius: 8px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.07);
                border: 1px solid var(--border-color);
                text-align: center; /* Center image inside */
            }
            .chart-container img {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                /* box-shadow: 0 1px 3px rgba(0,0,0,0.1); Removed inner shadow, outer container has shadow */
            }
             .chart-container h4 {
                 margin-top: 0;
                 margin-bottom: 15px;
                 text-align: center;
                 color: var(--text-color);
             }

            /* --- Tabs --- */
            .tabs {
                display: flex;
                flex-wrap: wrap; /* Allow wrapping on smaller screens */
                background-color: var(--medium-gray);
                border-radius: 6px 6px 0 0;
                border: 1px solid var(--border-color);
                border-bottom: none; /* Bottom border handled by content */
                margin-bottom: 0;
            }
            .tab-button {
                background-color: inherit;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 20px;
                transition: background-color 0.3s ease, color 0.3s ease;
                font-size: 1rem;
                font-family: var(--font-chinese), var(--font-sans-serif);
                flex-grow: 1; /* Allow buttons to grow */
                text-align: center;
                color: var(--dark-gray);
                border-right: 1px solid var(--border-color);
                font-weight: 500;
            }
            .tab-button:last-child { border-right: none; }
            .tab-button:hover { background-color: #d1d5db; color: var(--text-color); }
            .tab-button.active {
                background-color: var(--white);
                color: var(--primary-color);
                border-bottom: 3px solid var(--primary-color); /* Indicator line */
                font-weight: 600;
                position: relative;
                top: 1px; /* Align with content border */
            }
            .tab-content {
                display: none;
                padding: 25px;
                border: 1px solid var(--border-color);
                border-top: none;
                border-radius: 0 0 6px 6px;
                background-color: var(--white);
            }
            .tab-content.active { display: block; } /* JS will handle adding this class */

            /* --- Case Analysis --- */
            .case-analysis-section h4 {
                 margin-top: 0;
                 margin-bottom: 5px;
                 font-size: 1.3rem;
                 color: var(--text-color);
             }
             .case-analysis-info {
                 font-size: 0.9rem;
                 color: var(--dark-gray);
                 margin-bottom: 20px;
             }

            .case-analysis-tabs { /* Similar style to main tabs */
                display: flex;
                background-color: var(--medium-gray);
                border-radius: 6px;
                border: 1px solid var(--border-color);
                margin: 20px 0;
                overflow: hidden; /* Contained border radius */
            }
            .case-analysis-tab-button {
                background-color: inherit;
                border: none; outline: none; cursor: pointer;
                padding: 10px 15px; /* Slightly smaller */
                transition: background-color 0.3s ease, color 0.3s ease;
                font-size: 0.95rem;
                font-family: var(--font-chinese), var(--font-sans-serif);
                flex: 1; text-align: center;
                color: var(--dark-gray);
                border-right: 1px solid var(--border-color);
                font-weight: 500;
            }
            .case-analysis-tab-button:last-child { border-right: none; }
            .case-analysis-tab-button:hover { background-color: #d1d5db; color: var(--text-color); }
            .case-analysis-tab-button.active {
                background-color: var(--white);
                color: var(--secondary-color); /* Use secondary color for sub-tabs */
                font-weight: 600;
            }
            .case-analysis-content { display: none; }
             .case-analysis-content.active { display: block; } /* JS will handle adding this class */

            .cases-grid {
                display: grid;
                /* Modify to show only 2 per row */
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 25px;
                margin-top: 15px;
            }
            @media (max-width: 1250px) { .cases-grid { grid-template-columns: 1fr; } } /* Stack on smaller screens */


            .case-item {
                border: 1px solid var(--border-color);
                border-radius: 6px;
                background-color: var(--light-gray);
                padding: 15px;
                transition: box-shadow 0.2s ease;
            }
            .case-item:hover {
                 box-shadow: 0 4px 8px rgba(0,0,0,0.08);
             }

            .case-header {
                 display: flex;
                 justify-content: space-between;
                 align-items: center;
                 margin-bottom: 15px;
                 padding-bottom: 10px;
                 border-bottom: 1px solid var(--border-color);
                 flex-wrap: wrap; /* Wrap if needed */
             }
             .case-header h5 {
                 margin: 0;
                 font-size: 1.1rem;
                 font-weight: 600;
                 color: var(--text-color);
                 flex-basis: 60%; /* Allow space for metrics */
             }
             .full-metrics {
                 font-size: 0.85rem;
                 color: var(--dark-gray);
                 text-align: right;
                 flex-basis: 35%; /* Allow space */
                 white-space: nowrap; /* Prevent wrapping of metrics */
             }

            .case-comparison {
                 display: flex;
                 gap: 15px; /* Space between columns */
                 justify-content: space-between;
             }
             @media (max-width: 500px) { .case-comparison { flex-direction: column; } } /* Stack images vertically */


             .image-col {
                 flex: 1; /* Each column takes equal space */
                 text-align: center;
                 min-width: 0; /* Prevent overflow */
             }
             .image-col img {
                 width: 100%;
                 height: auto; /* Maintain aspect ratio */
                 max-height: 250px; /* Limit height */
                 object-fit: contain; /* Fit within bounds */
                 border: 1px solid var(--border-color);
                 border-radius: 4px;
                 background-color: var(--white); /* Background if image is transparent */
                 margin-bottom: 8px;
             }
             .image-label {
                 font-size: 0.9rem;
                 font-weight: 500;
                 margin-bottom: 5px;
                 color: var(--text-color);
             }
             .patch-metrics {
                 font-size: 0.85rem;
                 color: var(--dark-gray);
                 line-height: 1.4;
                 min-height: 2.8em; /* Reserve space for 2 lines */
             }
              .patch-metrics span { color: var(--text-color); } /* Make bold numbers stand out */


            /* --- Utility Classes --- */
            .error {
                color: var(--danger-dark);
                background-color: var(--danger-light);
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                border-left: 5px solid var(--danger-dark);
            }
            .info {
                 color: #0c5460;
                 background-color: #d1ecf1;
                 border-color: #bee5eb;
                 padding: 15px;
                 border-radius: 5px;
                 margin: 15px 0;
                 border-left: 5px solid #17a2b8;
             }

        </style>
        """

        # --- 用于标签和放大镜的JavaScript ---
        javascript = """
        <script>
            // Initialize tabs and setup magnifier functionality when document loads
            document.addEventListener('DOMContentLoaded', function() {
                // First initialize tabs
                let firstTabButton = document.querySelector('.tab-button');
                if (firstTabButton) {
                    firstTabButton.click();
                } else {
                    // Fallback if no tabs exist
                    let firstTabContent = document.querySelector('.tab-content');
                    if(firstTabContent) firstTabContent.style.display = 'block';
                }
                
                // Setup global magnifier
                setupGlobalMagnifier();
            });
        
            function openTab(evt, tabId) {
                // Handle main dataset tabs
                let tabcontent = document.querySelectorAll(".tab-content");
                tabcontent.forEach(tc => tc.style.display = "none");

                let tablinks = document.querySelectorAll(".tab-button");
                tablinks.forEach(tl => tl.classList.remove("active"));

                document.getElementById(tabId).style.display = "block";
                evt.currentTarget.classList.add("active");

                // Automatically activate the first case analysis tab within the opened dataset tab
                let firstCaseTabButton = document.querySelector(`#${tabId} .case-analysis-tab-button`);
                if (firstCaseTabButton) {
                    // Find the target ID from the onclick attribute
                     let onclickAttr = firstCaseTabButton.getAttribute('onclick');
                     let targetCaseTabId = onclickAttr.split("'")[1]; // Extracts the second argument like 'best-Set5-2'
                     // Simulate click if it's not already active - safer than direct style manipulation
                     if (!firstCaseTabButton.classList.contains('active')) {
                         openCaseAnalysisTab({ currentTarget: firstCaseTabButton }, targetCaseTabId);
                     } else {
                         // If it IS the active one, ensure its content is visible (in case switching back to the dataset tab)
                         let caseTabContent = document.getElementById(targetCaseTabId);
                         if (caseTabContent) caseTabContent.style.display = "block";
                     }
                }
            }

            function openCaseAnalysisTab(evt, tabId) {
                // Handle best/worst case tabs within a dataset tab
                let parentTabContent = evt.currentTarget.closest('.tab-content');
                if (!parentTabContent) return; // Should not happen

                let caseTabContent = parentTabContent.querySelectorAll(".case-analysis-content");
                caseTabContent.forEach(ctc => ctc.style.display = "none");

                let caseTabLinks = parentTabContent.querySelectorAll(".case-analysis-tab-button");
                caseTabLinks.forEach(ctl => ctl.classList.remove("active"));

                let targetContent = document.getElementById(tabId);
                 if (targetContent) {
                     targetContent.style.display = "block";
                 }
                evt.currentTarget.classList.add("active");
            }
            
            // Global magnifier functionality with centralized controls
            function setupGlobalMagnifier() {
                // Variables for magnifier state (global settings)
                let magnifierActive = false;
                let zoomLevel = 3; // Default zoom level
                let magnifierSize = 300; // Default size in pixels
                
                // Add global control panel to the page
                const controlPanelHTML = `
                    <div class="magnifier-control-panel">
                        <div class="magnifier-controls">
                            <div class="magnifier-toggle">
                                <input type="checkbox" id="global-magnifier-toggle" class="toggle-checkbox">
                                <label for="global-magnifier-toggle" class="toggle-label">打开放大镜</label>
                            </div>
                            <div class="magnifier-zoom">
                                <span class="zoom-label">缩放: <span id="global-zoom-value">${zoomLevel}x</span></span>
                                <input type="range" min="1.5" max="6" step="0.5" value="${zoomLevel}" id="global-zoom-slider" class="zoom-slider">
                            </div>
                            <div class="magnifier-size">
                                <span class="size-label">尺寸: <span id="global-size-value">${magnifierSize}px</span></span>
                                <input type="range" min="150" max="400" step="10" value="${magnifierSize}" id="global-size-slider" class="size-slider">
                            </div>
                        </div>
                    </div>
                `;
                
                // Add the control panel after the header section
                const reportHeader = document.querySelector('.report-header');
                if (reportHeader) {
                    reportHeader.insertAdjacentHTML('afterend', controlPanelHTML);
                } else {
                    // Fallback to beginning of container
                    const container = document.querySelector('.container');
                    if (container) {
                        container.insertAdjacentHTML('afterbegin', controlPanelHTML);
                    }
                }
                
                // Get references to the global controls
                const toggleInput = document.getElementById('global-magnifier-toggle');
                const zoomSlider = document.getElementById('global-zoom-slider');
                const zoomValue = document.getElementById('global-zoom-value');
                const sizeSlider = document.getElementById('global-size-slider');
                const sizeValue = document.getElementById('global-size-value');
                
                // Initialize magnifiers for all image comparisons (including in hidden tabs)
                initializeAllMagnifiers();
                
                // Function to initialize magnifiers for all image comparisons
                function initializeAllMagnifiers() {
                    // Find all image comparison containers across all tabs
                    const comparisons = document.querySelectorAll('.case-comparison');
                    
                    comparisons.forEach(comparison => {
                        // Get all images in this comparison
                        const images = comparison.querySelectorAll('img');
                        
                        // Create magnifiers for each image
                        images.forEach(img => {
                            // Check if magnifier already exists for this image
                            const parentCol = img.closest('.image-col');
                            if (parentCol && !parentCol.querySelector('.magnifier')) {
                                // Create the magnifier overlay
                                const magnifier = document.createElement('div');
                                magnifier.className = 'magnifier';
                                magnifier.style.width = `${magnifierSize}px`;
                                magnifier.style.height = `${magnifierSize}px`;
                                magnifier.style.display = 'none';
                                
                                // Position magnifier container properly
                                const imageCol = img.closest('.image-col');
                                imageCol.style.position = 'relative';
                                imageCol.appendChild(magnifier);
                                
                                // Set up event listeners for mouse movement
                                img.addEventListener('mousemove', e => {
                                    if (!magnifierActive) return;
                                    
                                    // Get cursor position relative to image
                                    const rect = img.getBoundingClientRect();
                                    const x = e.clientX - rect.left;
                                    const y = e.clientY - rect.top;
                                    
                                    // Update the magnifier
                                    updateMagnifier(magnifier, img, x, y, zoomLevel, magnifierSize);
                                    
                                    // Synchronize other magnifiers in the same comparison
                                    syncMagnifiers(comparison, img, x, y, rect.width, rect.height);
                                });
                                
                                // Show magnifier when mouse enters image
                                img.addEventListener('mouseenter', () => {
                                    if (magnifierActive) {
                                        magnifier.style.display = 'block';
                                    }
                                });
                                
                                // Hide magnifier when mouse leaves image
                                img.addEventListener('mouseleave', () => {
                                    magnifier.style.display = 'none';
                                });
                            }
                        });
                    });
                }
                
                // Global toggle magnifier on/off
                toggleInput.addEventListener('change', () => {
                    magnifierActive = toggleInput.checked;
                    
                    // No need to update display - it will update on mouse enter/leave
                });
                
                // Update global zoom level
                zoomSlider.addEventListener('input', () => {
                    zoomLevel = parseFloat(zoomSlider.value);
                    zoomValue.textContent = `${zoomLevel}x`;
                    
                    // Update all visible magnifiers
                    updateAllVisibleMagnifiers();
                });
                
                // Update global magnifier size
                sizeSlider.addEventListener('input', () => {
                    magnifierSize = parseInt(sizeSlider.value);
                    sizeValue.textContent = `${magnifierSize}px`;
                    
                    // Update all magnifiers size
                    document.querySelectorAll('.magnifier').forEach(mag => {
                        mag.style.width = `${magnifierSize}px`;
                        mag.style.height = `${magnifierSize}px`;
                    });
                    
                    // Update the positions and backgrounds
                    updateAllVisibleMagnifiers();
                });
                
                // Function to update all visible magnifiers across all tabs/cases
                function updateAllVisibleMagnifiers() {
                    document.querySelectorAll('.magnifier').forEach(magnifier => {
                        if (magnifier.style.display === 'block') {
                            const img = magnifier.closest('.image-col').querySelector('img');
                            const rect = img.getBoundingClientRect();
                            const magRect = magnifier.getBoundingClientRect();
                            const x = (magRect.left + magRect.width/2) - rect.left;
                            const y = (magRect.top + magRect.height/2) - rect.top;
                            
                            updateMagnifier(magnifier, img, x, y, zoomLevel, magnifierSize);
                        }
                    });
                }
                
                // Function to synchronize magnifiers across images in a comparison
                function syncMagnifiers(comparison, activeImg, x, y, imgWidth, imgHeight) {
                    const images = comparison.querySelectorAll('img');
                    const magnifiers = comparison.querySelectorAll('.magnifier');
                    
                    // Convert coordinates to percentage
                    const xPercent = x / imgWidth;
                    const yPercent = y / imgHeight;
                    
                    // Apply to all images except the active one
                    images.forEach((img, index) => {
                        if (img !== activeImg) {
                            const imgCol = img.closest('.image-col');
                            const magnifier = imgCol.querySelector('.magnifier');
                            if (magnifier && magnifier.style.display === 'block') {
                                // Convert percentage to pixels for this image
                                const imgRect = img.getBoundingClientRect();
                                const syncX = xPercent * imgRect.width;
                                const syncY = yPercent * imgRect.height;
                                
                                updateMagnifier(magnifier, img, syncX, syncY, zoomLevel, magnifierSize);
                            }
                        }
                    });
                }
                
                // Function to update an individual magnifier
                function updateMagnifier(magnifier, img, x, y, zoom, size) {
                    // Get image dimensions
                    const imgWidth = img.offsetWidth;
                    const imgHeight = img.offsetHeight;
                    
                    // Calculate magnifier position (centered on cursor)
                    const halfSize = size / 2;
                    let left = x - halfSize;
                    let top = y - halfSize;
                    
                    // Keep magnifier within image bounds
                    left = Math.max(0, Math.min(left, imgWidth - size));
                    top = Math.max(0, Math.min(top, imgHeight - size));
                    
                    // Set magnifier position
                    magnifier.style.left = `${left}px`;
                    magnifier.style.top = `${top}px`;
                    
                    // Calculate the zoomed background position
                    const bgX = (x * zoom - halfSize);
                    const bgY = (y * zoom - halfSize);
                    
                    // Update the magnifier's background
                    magnifier.style.backgroundImage = `url(${img.src})`;
                    magnifier.style.backgroundSize = `${imgWidth * zoom}px ${imgHeight * zoom}px`;
                    magnifier.style.backgroundPosition = `-${bgX}px -${bgY}px`;
                }
                
                // Re-initialize magnifiers for newly displayed tab content
                // Add event listener to tab buttons to initialize magnifiers in newly opened tabs
                document.querySelectorAll('.tab-button').forEach(button => {
                    button.addEventListener('click', function() {
                        // Short delay to ensure tab content is visible
                        setTimeout(initializeAllMagnifiers, 100);
                    });
                });
                
                // Also reinitialize when case analysis tabs are clicked
                document.querySelectorAll('.case-analysis-tab-button').forEach(button => {
                    button.addEventListener('click', function() {
                        setTimeout(initializeAllMagnifiers, 100);
                    });
                });
            }
        </script>
        """

        # --- HTML主体生成 ---

        # 计算总体指标
        overall_psnr = []
        overall_ssim = []
        total_samples = 0
        for dataset_name, scale_results in all_results.items():
            for scale, results in scale_results.items():
                if results:
                    psnr_vals = [r[1] for r in results if r[1] is not None]
                    ssim_vals = [r[2] for r in results if r[2] is not None]
                    overall_psnr.extend(psnr_vals)
                    overall_ssim.extend(ssim_vals)
                    total_samples += len(
                        results
                    )  # 计算所有尝试的数量，即使有些指标计算失败

        mean_overall_psnr = np.mean(overall_psnr) if overall_psnr else 0
        mean_overall_ssim = np.mean(overall_ssim) if overall_ssim else 0
        num_datasets = len(all_results)

        # 开始HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{model_name} - 超分辨率评估报告</title>
            {style}
        </head>
        <body>
            <div class="container">
                <div class="report-header">
                    <h1>{model_name} - 超分辨率评估报告</h1>
                    <p>对多个标准测试数据集的超分辨率结果进行量化评估和可视化分析。</p>
                </div>

                <div class="section">
                    <h2>总体摘要</h2>
                    <div class="summary-grid">
                        <div class="summary-card">
                            <h3>测试数据集</h3>
                            <span class="value">{num_datasets}</span>
                            <span class="unit">个</span>
                        </div>
                        <div class="summary-card">
                            <h3>总样本数</h3>
                            <span class="value">{total_samples}</span>
                             <span class="unit">张图像</span>
                        </div>
                        <div class="summary-card">
                            <h3>平均 PSNR <small>(Y通道)</small></h3>
                            <span class="value">{mean_overall_psnr:.2f}</span>
                             <span class="unit">dB</span>
                        </div>
                        <div class="summary-card">
                            <h3>平均 SSIM <small>(Y通道)</small></h3>
                            <span class="value">{mean_overall_ssim:.4f}</span>
                             <span class="unit"></span>
                        </div>
                    </div>

                    <h3>各数据集平均指标</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>数据集</th>
                                <th>缩放比例</th>
                                <th>有效样本数</th>
                                <th>平均 PSNR (dB)</th>
                                <th>平均 SSIM</th>
                            </tr>
                        </thead>
                        <tbody>
        """

        # 添加汇总表格行
        valid_sample_count_total = 0
        for dataset_name, scale_results in all_results.items():
            for scale, results in scale_results.items():
                if results:
                    valid_psnr = [r[1] for r in results if r[1] is not None]
                    valid_ssim = [r[2] for r in results if r[2] is not None]
                    valid_count = len(valid_psnr)  # 统计有效PSNR的样本数
                    valid_sample_count_total += valid_count
                    avg_psnr = np.mean(valid_psnr) if valid_psnr else 0
                    avg_ssim = np.mean(valid_ssim) if valid_ssim else 0

                    summary_table_row = f"<tr><td>{dataset_name}</td><td>x{scale}</td><td>{valid_count} / {len(results)}</td>"
                    summary_table_row += (
                        f"<td>{avg_psnr:.4f}</td><td>{avg_ssim:.4f}</td></tr>\n"
                    )
                    html_content += summary_table_row

        # 如果有有效结果，添加总行
        if valid_sample_count_total > 0:
            html_content += f'<tr class="total-row">'
            html_content += f'<td colspan="2">总平均 ({valid_sample_count_total} / {total_samples} 有效样本)</td><td>{valid_sample_count_total}</td>'
            html_content += f"<td>{mean_overall_psnr:.4f}</td><td>{mean_overall_ssim:.4f}</td></tr>\n"

        html_content += """
                        </tbody>
                    </table>
                </div>

                <div class="section">
                    <h2>数据集详细分析</h2>
                    <div class="tabs">
        """

        # 添加数据集选项卡按钮
        dataset_keys = list(all_results.keys())
        for i, dataset_name in enumerate(dataset_keys):
            active_class = "active" if i == 0 else ""
            html_content += f'<button class="tab-button {active_class}" onclick="openTab(event, \'{dataset_name}\')">{dataset_name}</button>\n'

        html_content += "</div>\n"  # End of .tabs

        # 添加每个数据集选项卡的内容
        for i, (dataset_name, scale_results) in enumerate(all_results.items()):
            display_style = (
                "block" if i == 0 else "none"
            )  # 现在由JS处理，但为了结构保留
            html_content += f'<div id="{dataset_name}" class="tab-content" style="display: {display_style};">\n'

            if not scale_results:
                html_content += (
                    f"<div class='info'>没有找到 {dataset_name} 的结果。</div>"
                )
                html_content += "</div>\n"  # 关闭tab-content
                continue

            # 假设每个数据集在此结构中只有一个scale，以简化处理
            # 如果每个数据集需要多个scale，则添加内部选项卡或部分
            for scale, results in scale_results.items():
                html_content += f"<h3>{dataset_name} - x{scale} 放大</h3>\n"

                if results:
                    dataset_dir = os.path.join(dataset_path, dataset_name)
                    results_dir = os.path.join(output_path, dataset_name)

                    # 生成分布图并转换为base64
                    try:
                        # 创建分布图
                        dist_fig = Visualizer.plot_dataset_results(
                            results, dataset_name, scale, figsize=(12, 7), dpi=100
                        )

                        # 将图转换为base64字符串
                        if dist_fig:
                            chart_base64 = ImageUtils.plt_figure_to_base64(dist_fig)

                            # 添加嵌入式图表
                            html_content += f"""
                            <div class="chart-container">
                                <h4>指标分布图</h4>
                                <img src="data:image/png;base64,{chart_base64}" alt="{dataset_name} x{scale} 指标分布">
                            </div>
                            """
                        else:
                            html_content += f"<div class='info'>无法为 {dataset_name} 生成分布图表。</div>"
                    except Exception as e:
                        html_content += f"<div class='info'>生成 {dataset_name} 分布图时出错: {str(e)}</div>"

                    # 案例分析部分
                    html_content += f"""
                    <div class="case-analysis-container">
                        <h4>典型案例分析</h4>
                        <div class="case-analysis-tabs">
                            <button class="case-analysis-tab-button active" onclick="openCaseAnalysisTab(event, 'best-{dataset_name}-{scale}')">最佳案例 (按PSNR)</button>
                            <button class="case-analysis-tab-button" onclick="openCaseAnalysisTab(event, 'worst-{dataset_name}-{scale}')">最差案例 (按PSNR)</button>
                        </div>

                        <div id="best-{dataset_name}-{scale}" class="case-analysis-content active">
                            {Visualizer.generate_cases_analysis_html(results_dir, dataset_dir, results, scale, "best", patch_size, interp_method, num_cases)}
                        </div>
                        <div id="worst-{dataset_name}-{scale}" class="case-analysis-content">
                            {Visualizer.generate_cases_analysis_html(results_dir, dataset_dir, results, scale, "worst", patch_size, interp_method, num_cases)}
                        </div>
                    </div>
                    """
                else:
                    html_content += f"<div class='info'>数据集 {dataset_name} (x{scale}) 没有有效的评估结果。</div>"

            html_content += "</div>\n"  # 关闭tab-content

        html_content += """
                </div>
            </div>
            {javascript}
        </body>
        </html>
        """.format(javascript=javascript)

        # 保存HTML报告
        report_filepath = os.path.join(output_path, "report.html")
        try:
            os.makedirs(os.path.join(output_path, "charts"), exist_ok=True)
            with open(report_filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"HTML报告已成功生成: {report_filepath}")
        except Exception as e:
            print(f"错误：无法写入HTML报告到 {report_filepath}: {e}")


# ===============================
# 5. 模型加载和推理模块
# ===============================
class ModelHandler:
    """模型加载和推理处理类"""

    @staticmethod
    def load_model(arch, model_path, scale=None):
        """根据架构名加载模型

        Args:
            arch: 模型架构名称（大写英语）
            model_path: 模型权重文件路径
            scale: 超分辨率缩放因子（某些模型需要）

        Returns:
            加载好权重的模型
        """
        try:
            # 动态导入模块，模块名为小写形式的 arch
            module_name = f"tools.models.{arch.lower()}"
            module = __import__(module_name, fromlist=[arch])

            # 获取模型类并实例化
            model_class = getattr(module, arch)
            model = model_class()

            # 加载模型权重
            # 首先尝试 load_model 函数（如果存在），否则直接加载状态字典
            try:
                from tools.utils import load_model as utils_load_model

                model = utils_load_model(model, model_path)
            except (ImportError, AttributeError):
                model.load_state_dict(torch.load(model_path, map_location="cpu"))

            return model
        except ImportError:
            raise ValueError(f"不支持的模型架构或模块导入失败: {arch}")
        except Exception as e:
            raise RuntimeError(f"加载模型 {arch} 失败: {e}") from e

    @staticmethod
    def run_inference(model, lr_path, output_path, device="cuda:0"):
        """在低分辨率图像上运行模型推理

        Args:
            model: 已加载的模型
            lr_path: 低分辨率图像目录
            output_path: 输出目录
            device: 运行设备

        Returns:
            None，结果保存到输出目录
        """
        try:
            from tools.utils import infer_from_model, patch_infer_from_model

            patch_infer_from_model(model, lr_path, output_path, device=device)
            return True
        except ImportError:
            raise ImportError(
                "无法导入 tools.utils.infer_from_model 函数，请确保它存在"
            )
        except Exception as e:
            print(f"推理过程中出错: {e}")
            return False


# ===============================
# 6. 评估配置类
# ===============================
class SREvalConfig:
    """超分辨率评估配置类"""

    def __init__(
        self,
        model_path=None,
        root_dir="SR_Test",
        output_dir="SR_Results",
        scale=2,
        device="cuda:0",
        patch_size=256,
        interp_method="bicubic",
        num_cases=8,
        num_workers=None,
        arch="SMFANET",
        chart_dpi=150,
        chart_width=16,
        chart_height=10,
    ):
        self.model_path = model_path
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.scale = scale
        self.device = device
        self.patch_size = patch_size
        self.interp_method = interp_method
        self.num_cases = num_cases
        self.num_workers = num_workers
        self.arch = arch
        self.chart_dpi = chart_dpi
        self.chart_width = chart_width
        self.chart_height = chart_height

        # 通过model_path提取模型名称（如果有）
        self.model_name = self._extract_model_name()

    def _extract_model_name(self):
        """从模型路径提取模型名称"""
        if not self.model_path:
            return self.arch

        try:
            # 尝试从路径获取模型名
            path_parts = self.model_path.split("/")
            if len(path_parts) > 1:
                return path_parts[1]  # 假设路径格式为"checkpoints/MODEL_NAME.pth"
            else:
                return os.path.splitext(os.path.basename(self.model_path))[0]
        except:
            return self.arch

    def get_effective_output_dir(self):
        """获取包含模型名和缩放因子的有效输出目录"""
        return os.path.join(self.output_dir, f"{self.model_name}_x{self.scale}")

    @classmethod
    def from_args(cls, args):
        """从argparse.Namespace创建配置"""
        return cls(
            model_path=args.model_path,
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            scale=args.scale,
            device=args.device,
            patch_size=args.patch_size,
            interp_method=args.interp_method,
            num_cases=args.num_cases,
            num_workers=args.num_workers,
            arch=args.arch,
            chart_dpi=args.chart_dpi,
            chart_width=args.chart_width,
            chart_height=args.chart_height,
        )


# ===============================
# 7. 评估流程函数
# ===============================
def evaluate_sr_model(config):
    """评估超分辨率模型在多个数据集上的性能

    Args:
        config: SREvalConfig实例

    Returns:
        all_results: 所有数据集的评估结果
    """
    # 创建输出目录
    effective_output_dir = config.get_effective_output_dir()
    os.makedirs(effective_output_dir, exist_ok=True)
    print(f"结果将保存到: {effective_output_dir}")

    # 加载模型
    print(f"正在加载模型架构: {config.arch}")
    try:
        model = ModelHandler.load_model(config.arch, config.model_path, config.scale)
        model = model.to(config.device)
        model.eval()
        print("模型加载成功.")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

    # 查找可用数据集
    available_datasets = ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]
    datasets_to_process = []
    print(f"正在根目录 '{config.root_dir}' 中查找数据集...")

    for d in available_datasets:
        dataset_path = os.path.join(config.root_dir, d)
        hr_dir = os.path.join(dataset_path, "HR")
        lr_dir = os.path.join(dataset_path, f"LR_bicubic/X{config.scale}")  # 常见结构
        lr_dir_alt = os.path.join(dataset_path, f"x{config.scale}")  # 备选结构

        if (
            os.path.exists(dataset_path)
            and os.path.exists(hr_dir)
            and (os.path.exists(lr_dir) or os.path.exists(lr_dir_alt))
        ):
            print(f"  找到: {d}")
            datasets_to_process.append(d)

    if not datasets_to_process:
        print(
            f"错误: 在 '{config.root_dir}' 中未找到任何完整的数据集 (需要 HR 和 x{config.scale} 或 LR_bicubic/X{config.scale} 子目录)."
        )
        return None

    # 处理每个数据集
    all_results = {}
    for dataset in datasets_to_process:
        print("-" * 50)
        print(f"处理数据集: {dataset} (x{config.scale})")
        dataset_path = os.path.join(config.root_dir, dataset)

        # 确定LR路径（处理两种常见命名约定）
        lr_path = os.path.join(dataset_path, f"LR_bicubic/X{config.scale}")
        if not os.path.exists(lr_path):
            lr_path = os.path.join(dataset_path, f"x{config.scale}")
            if not os.path.exists(lr_path):
                print(
                    f"  错误: 找不到 {dataset} 的 LR 图像目录 (尝试了 LR_bicubic/X{config.scale} 和 x{config.scale})"
                )
                continue

        # 预测输出路径
        dataset_output_dir = os.path.join(effective_output_dir, dataset)
        pred_path = os.path.join(dataset_output_dir, f"PRED_x{config.scale}")
        os.makedirs(pred_path, exist_ok=True)
        print(f"  LR 路径: {lr_path}")
        print(f"  Pred 路径: {pred_path}")

        # 运行推理
        print(f"  正在对 {dataset} 进行推理...")
        try:
            success = ModelHandler.run_inference(
                model, lr_path, pred_path, device=config.device
            )
            if success:
                print("  推理完成.")
            else:
                print("  推理失败，跳过评估.")
                continue
        except Exception as e:
            print(f"  推理过程中出错: {e}")
            continue  # 如果推理失败，跳过评估

        # 评估结果
        print(f"  正在评估 {dataset} 的推理结果...")
        results = MetricsCalculator.evaluate_sr_results(
            dataset_path, pred_path, config.scale, config.num_workers
        )

        if results and len(results) > 0:
            if dataset not in all_results:
                all_results[dataset] = {}
            all_results[dataset][config.scale] = results

            # 打印指标摘要
            psnr_values = [r[1] for r in results if r[1] is not None]
            ssim_values = [r[2] for r in results if r[2] is not None]
            if psnr_values and ssim_values:  # 只有在有有效指标时才打印
                print(f"  {dataset} (x{config.scale}) 结果:")
                print(f"    有效样本: {len(psnr_values)} / {len(results)}")
                print(f"    平均 PSNR: {np.mean(psnr_values):.4f} dB")
                print(f"    平均 SSIM: {np.mean(ssim_values):.4f}")
                print(
                    f"    最好 PSNR: {np.max(psnr_values):.4f} dB / 最差 PSNR: {np.min(psnr_values):.4f} dB"
                )
            else:
                print(f"  {dataset} (x{config.scale}): 计算指标时出错或无有效结果。")

            # 创建并保存可视化图表
            print(f"  创建 {dataset} 的指标分布图...")
            dist_fig = Visualizer.plot_dataset_results(
                results,
                dataset,
                config.scale,
                figsize=(config.chart_width, config.chart_height),
                dpi=config.chart_dpi,
            )

            if dist_fig:
                chart_dir = os.path.join(effective_output_dir, "charts")
                os.makedirs(chart_dir, exist_ok=True)
                chart_filepath = os.path.join(
                    chart_dir, f"{dataset}_x{config.scale}_distribution.png"
                )
                try:
                    dist_fig.savefig(
                        chart_filepath, dpi=config.chart_dpi, bbox_inches="tight"
                    )
                    print(f"  图表已保存: {chart_filepath}")
                except Exception as e:
                    print(f"  保存图表时出错: {e}")
                finally:
                    plt.close(dist_fig)  # 关闭图表以释放内存
        else:
            print(f"  未能为 {dataset} (x{config.scale}) 获取有效的评估结果。")

    return all_results, effective_output_dir


# ===============================
# 8. 参数解析函数
# ===============================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SR图像推理、评估和可视化")

    # 基本参数
    parser.add_argument(
        "--model_path", type=str, required=False, help="模型权重文件路径 (.pth)"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="SR_Test",
        help="包含各数据集文件夹 (如 Set5, Urban100) 的根目录",
    )
    parser.add_argument(
        "--scale", type=int, default=2, choices=[2, 3, 4], help="超分辨率比例因子"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="使用的设备 (e.g., 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--output_dir", type=str, default="SR_Results", help="结果输出根目录"
    )

    # 可配置参数
    parser.add_argument(
        "--patch_size", type=int, default=128, help="案例分析中用于比较的中心区域边长"
    )
    parser.add_argument(
        "--interp_method",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="用于比较的插值方法",
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=6,
        help="每个数据集显示的最佳/最差案例数量 (建议 6-12)",
    )
    parser.add_argument("--chart_dpi", type=int, default=300, help="保存图表的DPI")
    parser.add_argument("--chart_width", type=int, default=16, help="图表宽度(英寸)")
    parser.add_argument("--chart_height", type=int, default=10, help="图表高度(英寸)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="评估指标时并行处理的工作线程数 (默认: min(CPU核心数, 8))",
    )

    # 模型选择参数
    parser.add_argument(
        "--arch",
        type=str,
        default="MYNET",
        help="要加载的模型架构",
    )

    return parser.parse_args()


# ===============================
# 9. 主程序入口
# ===============================
def main():
    """主程序入口"""
    # 解析参数
    args = parse_args()

    # 创建配置
    config = SREvalConfig.from_args(args)

    # 评估模型
    all_results, effective_output_dir = evaluate_sr_model(config)

    # 生成HTML报告
    if all_results:
        print("-" * 50)
        print("正在生成最终HTML报告...")

        patch_tuple = (config.patch_size, config.patch_size)
        ReportGenerator.create_html_report(
            all_results,
            effective_output_dir,  # 在特定运行的目录中保存报告
            config.root_dir,
            config.model_name,
            patch_tuple,
            config.interp_method,
            config.num_cases,
        )
        print(f"评估完成！结果和报告已保存到 {effective_output_dir}")
    else:
        print("没有足够的数据来生成报告。请检查数据集路径和推理/评估过程。")


if __name__ == "__main__":
    main()
