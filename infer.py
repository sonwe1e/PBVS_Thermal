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
from tools.utils import load_model, infer_from_model
import base64
from io import BytesIO
import torch


# ===============================
# 1. 指标计算模块
# ===============================
class MetricsCalculator:
    @staticmethod
    def get_y_channel(img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img_ycbcr = img.convert("YCbCr")
        img_y, _, _ = img_ycbcr.split()
        return np.array(img_y)

    @staticmethod
    def calculate_metrics_y_channel(gt_img, pred_img):
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim

        # 确保两个图像大小一致，如不一致则调整预测图像大小
        if gt_img.size != pred_img.size:
            print(
                f"警告: 图像大小不一致 - GT: {gt_img.size}, Pred: {pred_img.size}，正在调整大小"
            )
            pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

        gt_y = MetricsCalculator.get_y_channel(gt_img)
        pred_y = MetricsCalculator.get_y_channel(pred_img)

        # 再次确认两个Y通道的大小是否一致
        if gt_y.shape != pred_y.shape:
            raise ValueError(
                f"Y通道大小不一致 - GT: {gt_y.shape}, Pred: {pred_y.shape}"
            )

        psnr_value = psnr(gt_y, pred_y, data_range=255)
        ssim_value = ssim(gt_y, pred_y, data_range=255)
        return psnr_value, ssim_value

    @staticmethod
    def process_single_image(args):
        img_path, hr_dir, pred_dir = args
        img_name = os.path.basename(img_path)
        hr_path = os.path.join(hr_dir, img_name)
        try:
            hr_img = Image.open(hr_path).convert("RGB")
            pred_img = Image.open(img_path).convert("RGB")

            # 确保两个图像大小一致，特别针对Urban100数据集
            if hr_img.size != pred_img.size:
                print(f"调整图像大小 {img_name}: {pred_img.size} -> {hr_img.size}")
                pred_img = pred_img.resize(hr_img.size, Image.BICUBIC)

            psnr_value, ssim_value = MetricsCalculator.calculate_metrics_y_channel(
                hr_img, pred_img
            )
            return img_name, psnr_value, ssim_value, img_name.split(".")[0]
        except Exception as e:
            print(f"处理 {img_path} 出错: {e}")
            return img_name, None, None, img_name.split(".")[0]

    @staticmethod
    def evaluate_sr_results(dataset_dir, dataset_name, scale, num_workers=None):
        """评估超分辨率结果"""
        hr_dir = os.path.join(dataset_dir, "HR")
        pred_dir = os.path.join(dataset_dir, f"PRED_x{scale}")

        if not os.path.exists(pred_dir):
            print(f"预测目录不存在: {pred_dir}")
            return None

        pred_files = glob.glob(os.path.join(pred_dir, "*.png"))
        if not pred_files:
            print(f"在 {pred_dir} 中未找到预测图像")
            return None

        results = []
        args_list = [(pred_path, hr_dir, pred_dir) for pred_path in pred_files]

        # 如果未指定工作线程数量，则使用CPU核心数
        if num_workers is None:
            num_workers = os.cpu_count()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for result in tqdm(
                executor.map(MetricsCalculator.process_single_image, args_list),
                total=len(args_list),
                desc=f"评估 {dataset_name} x{scale}",
            ):
                if result[1] is not None:
                    results.append(result)

        return results


# ===============================
# 2. 可视化模块
# ===============================
class Visualizer:
    @staticmethod
    def upscale_with_interpolation(lr_img, hr_size, method="bicubic"):
        """使用指定的插值方法放大图像"""
        interpolation_methods = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }

        interp_method = interpolation_methods.get(method.lower(), Image.BICUBIC)
        return lr_img.resize(hr_size, interp_method)

    @staticmethod
    def img_to_base64(img):
        """将PIL图像转换为base64字符串"""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    @staticmethod
    def plot_dataset_results(
        dataset_results, dataset_name, scale, figsize=(10, 6), dpi=100
    ):
        """为数据集结果创建可视化图表"""
        if not dataset_results or len(dataset_results) == 0:
            print(f"无法为 {dataset_name} 绘制图表：没有有效结果")
            return None

        psnr_values = [r[1] for r in dataset_results]
        ssim_values = [r[2] for r in dataset_results]

        # 创建图表
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(2, 2, figure=fig)

        # 配置字体
        plt.rcParams.update({"font.size": 9})

        # PSNR分布直方图
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(psnr_values, kde=True, ax=ax1, bins=15)
        ax1.axvline(np.mean(psnr_values), color="r", linestyle="--")
        ax1.set_title(f"{dataset_name} PSNR Distribution (x{scale})")
        ax1.set_xlabel("PSNR (dB)")
        ax1.text(
            0.95,
            0.95,
            f"Mean: {np.mean(psnr_values):.2f} dB",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            fontsize=8,
        )

        # SSIM分布直方图
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(ssim_values, kde=True, ax=ax2, bins=15)
        ax2.axvline(np.mean(ssim_values), color="r", linestyle="--")
        ax2.set_title(f"{dataset_name} SSIM Distribution (x{scale})")
        ax2.set_xlabel("SSIM")
        ax2.text(
            0.95,
            0.95,
            f"Mean: {np.mean(ssim_values):.4f}",
            transform=ax2.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            fontsize=8,
        )

        # PSNR vs SSIM散点图
        ax3 = fig.add_subplot(gs[1, :])
        scatter = ax3.scatter(
            psnr_values, ssim_values, alpha=0.6, c=psnr_values, cmap="viridis", s=30
        )
        ax3.set_title(f"{dataset_name} PSNR vs SSIM (x{scale})")
        ax3.set_xlabel("PSNR (dB)")
        ax3.set_ylabel("SSIM")
        plt.colorbar(scatter, ax=ax3, label="PSNR (dB)")

        plt.tight_layout()
        return fig

    @staticmethod
    def extract_center_patch(img, patch_size=(256, 256)):
        """从图像中心提取指定大小的区域并确保统一尺寸"""
        width, height = img.size
        left = (width - patch_size[0]) // 2
        top = (height - patch_size[1]) // 2
        right = left + patch_size[0]
        bottom = top + patch_size[1]

        # 确保裁剪区域在图像范围内
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)

        # 裁剪中心区域
        cropped = img.crop((left, top, right, bottom))

        # 如果裁剪区域与目标尺寸不同，调整为目标尺寸
        if cropped.size != patch_size:
            cropped = cropped.resize(patch_size, Image.BICUBIC)

        return cropped

    @staticmethod
    def generate_cases_analysis_html(
        dataset_dir,
        results,
        scale,
        base_path,
        case_type="best",
        patch_size=(256, 256),
        interp_method="bicubic",
        num_cases=12,
    ):
        """为最好或最差的案例生成HTML"""
        if not results or len(results) < num_cases:
            return f"<div>不足够的图像进行{case_type}案例分析</div>"

        # 排序找出最好或最差的指定数量图片
        if case_type == "best":
            # 按PSNR排序找出最好的图
            selected_cases = sorted(results, key=lambda x: x[1], reverse=True)[
                :num_cases
            ]
            title = f"最佳案例分析 (中心{patch_size[0]}x{patch_size[1]}区域)"
        else:
            # 按PSNR排序找出最差的图
            selected_cases = sorted(results, key=lambda x: x[1])[:num_cases]
            title = f"最差案例分析 (中心{patch_size[0]}x{patch_size[1]}区域)"

        html_content = f"""
        <div class="case-analysis-section">
            <h3>{title}</h3>
            <div class="cases-grid">
        """

        # 每两张图一行，每行包含两组图像（每组三列）
        for i in range(0, len(selected_cases), 2):
            html_content += '<div class="cases-row">'

            # 处理当前行的两张图像
            for j in range(2):
                if i + j < len(selected_cases):
                    img_name, psnr_val, ssim_val, _ = selected_cases[i + j]
                    try:
                        hr_path = os.path.join(dataset_dir, "HR", img_name)
                        sr_path = os.path.join(dataset_dir, f"PRED_x{scale}", img_name)
                        lr_path = os.path.join(dataset_dir, f"x{scale}", img_name)

                        hr_img = Image.open(hr_path).convert("RGB")
                        sr_img = Image.open(sr_path).convert("RGB")
                        lr_img = Image.open(lr_path).convert("RGB")

                        # 确保所有图像大小一致 - 以HR图像为标准
                        hr_size = hr_img.size
                        if sr_img.size != hr_size:
                            sr_img = sr_img.resize(hr_size, Image.BICUBIC)

                        # 对低分辨率图像进行插值放大
                        lr_upscaled = Visualizer.upscale_with_interpolation(
                            lr_img, hr_size, interp_method
                        )

                        # 计算全图指标 - 确保所有图像大小一致后计算
                        full_psnr, full_ssim = (
                            MetricsCalculator.calculate_metrics_y_channel(
                                hr_img, sr_img
                            )
                        )

                        # 提取中心指定大小区域，并保证尺寸统一
                        hr_center = Visualizer.extract_center_patch(hr_img, patch_size)
                        sr_center = Visualizer.extract_center_patch(sr_img, patch_size)
                        interp_center = Visualizer.extract_center_patch(
                            lr_upscaled, patch_size
                        )

                        # 确保所有图像都是相同的尺寸
                        assert hr_center.size == patch_size, (
                            f"HR center patch size mismatch: {hr_center.size} vs {patch_size}"
                        )
                        assert sr_center.size == patch_size, (
                            f"SR center patch size mismatch: {sr_center.size} vs {patch_size}"
                        )
                        assert interp_center.size == patch_size, (
                            f"Interpolated center patch size mismatch: {interp_center.size} vs {patch_size}"
                        )

                        # 计算中心区域指标
                        center_psnr, center_ssim = (
                            MetricsCalculator.calculate_metrics_y_channel(
                                hr_center, sr_center
                            )
                        )

                        interp_method_display = interp_method.capitalize()

                        html_content += f"""
                        <div class="case-group">
                            <div class="case-info">
                                <h4>{i + j + 1}. {img_name}</h4>
                                <div class="metrics">
                                    <span class="metric">PSNR: {full_psnr:.2f}dB</span>
                                    <span class="metric">SSIM: {full_ssim:.4f}</span>
                                </div>
                            </div>
                            <div class="case-images">
                                <div class="case-image-cell">
                                    <img src="data:image/png;base64,{Visualizer.img_to_base64(interp_center)}" alt="{interp_method_display}插值中心">
                                    <div class="image-type">{interp_method_display}插值</div>
                                </div>
                                <div class="case-image-cell">
                                    <img src="data:image/png;base64,{Visualizer.img_to_base64(sr_center)}" alt="超分辨率中心">
                                    <div class="image-type">超分辨率</div>
                                </div>
                                <div class="case-image-cell">
                                    <img src="data:image/png;base64,{Visualizer.img_to_base64(hr_center)}" alt="高分辨率中心">
                                    <div class="image-type">高分辨率</div>
                                </div>
                            </div>
                        </div>
                        """
                    except Exception as e:
                        html_content += (
                            f"<div class='error'>处理{img_name}时出错: {e}</div>"
                        )

            html_content += "</div>"

        html_content += """
            </div>
        </div>
        """
        return html_content


# ===============================
# 3. HTML报告生成模块
# ===============================
class ReportGenerator:
    @staticmethod
    def create_html_report(
        all_results,
        output_path,
        root_dir,
        patch_size=(256, 256),
        interp_method="bicubic",
        num_cases=12,
    ):
        """生成HTML评估报告"""
        # CSS样式定义
        style = """
        <style>
            body { 
                font-family: "Microsoft YaHei", "SimHei", sans-serif; 
                max-width: 90%; 
                margin: 0 auto; 
                padding: 20px;
                background-color: #f5f5f5; 
            }
            h1, h2, h3, h4 { color: #333; }
            table { 
                border-collapse: collapse; 
                width: 100%; 
                margin: 20px 0; 
                background-color: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            th, td { 
                border: 1px solid #ddd; 
                padding: 10px; 
                text-align: center; 
            }
            th { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold;
            }
            tr:nth-child(even) { background-color: #f9f9f9; }
            tr:hover { background-color: #f1f1f1; }
            .header { 
                background-color: #4CAF50; 
                color: white; 
                padding: 20px; 
                margin-bottom: 30px; 
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            .section { 
                margin: 30px 0; 
                background-color: white; 
                padding: 25px; 
                border-radius: 5px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .summary { 
                display: flex; 
                justify-content: space-around; 
                flex-wrap: wrap; 
                gap: 20px;
            }
            .summary-card { 
                background-color: white; 
                padding: 20px; 
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                flex: 1;
                min-width: 200px;
                max-width: 250px;
                text-align: center;
                transition: transform 0.2s;
            }
            .summary-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .summary-card h3 { 
                margin-top: 0; 
                color: #4CAF50; 
                font-size: 18px;
            }
            .value { 
                font-size: 28px; 
                font-weight: bold; 
                margin: 15px 0; 
                color: #333;
            }
            .chart { 
                width: 100%; 
                margin: 25px 0; 
                text-align: center; 
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .chart img { 
                max-width: 100%; 
                height: auto; 
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border-radius: 3px;
            }
            .tabs { 
                overflow: hidden; 
                background-color: #f1f1f1; 
                border-radius: 5px 5px 0 0; 
                display: flex;
                border: 1px solid #ddd;
                margin-bottom: 0;
            }
            .tab-button { 
                background-color: inherit; 
                border: none; 
                outline: none;
                cursor: pointer; 
                padding: 14px 20px; 
                transition: 0.3s; 
                font-size: 16px;
                font-family: "Microsoft YaHei", "SimHei", sans-serif;
                flex: 1;
                text-align: center;
                border-right: 1px solid #ddd;
                font-weight: bold;
            }
            .tab-button:last-child {
                border-right: none;
            }
            .tab-button:hover { 
                background-color: #ddd; 
            }
            .tab-button.active { 
                background-color: #4CAF50; 
                color: white;
            }
            .tab-content { 
                display: none; 
                padding: 25px; 
                border: 1px solid #ddd; 
                border-top: none; 
                border-radius: 0 0 5px 5px;
                background-color: white;
            }
            .metrics { 
                margin: 10px 0 20px; 
                font-size: 16px;
            }
            .metric { 
                margin-right: 20px; 
                background-color: #f2f2f2; 
                padding: 8px 15px; 
                border-radius: 20px;
                font-weight: bold;
                display: inline-block;
            }
            .error {
                color: #d32f2f;
                background-color: #ffebee;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                border-left: 5px solid #d32f2f;
            }
            /* 案例分析部分样式 */
            .case-analysis-section {
                margin: 30px 0;
                background-color: white;
                padding: 25px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .cases-grid {
                display: flex;
                flex-direction: column;
                gap: 30px;
                margin-top: 20px;
            }
            .cases-row {
                display: flex;
                gap: 20px;
                justify-content: space-between;
            }
            .case-group {
                width: 48%;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: #f9f9f9;
            }
            .case-info {
                margin-bottom: 10px;
            }
            .case-info h4 {
                margin: 0 0 5px 0;
                font-size: 14px;
            }
            .case-images {
                display: flex;
                gap: 5px;
            }
            .case-image-cell {
                width: 33.333%;
                text-align: center;
            }
            .case-image-cell img {
                width: 100%;
                border: 1px solid #ddd;
                border-radius: 3px;
                height: auto;
                object-fit: cover; 
                aspect-ratio: 1; 
            }
            .image-type {
                font-size: 12px;
                margin-top: 5px;
                color: #666;
            }
            .case-analysis-tabs {
                overflow: hidden; 
                background-color: #f1f1f1; 
                border-radius: 5px; 
                display: flex;
                border: 1px solid #ddd;
                margin: 15px 0;
            }
            .case-analysis-tab-button {
                background-color: inherit; 
                border: none; 
                outline: none;
                cursor: pointer; 
                padding: 14px 20px; 
                transition: 0.3s; 
                font-size: 16px;
                font-family: "Microsoft YaHei", "SimHei", sans-serif;
                flex: 1;
                text-align: center;
                border-right: 1px solid #ddd;
                font-weight: bold;
            }
            .case-analysis-tab-button:last-child {
                border-right: none;
            }
            .case-analysis-tab-button:hover { 
                background-color: #ddd; 
            }
            .case-analysis-tab-button.active { 
                background-color: #4CAF50; 
                color: white;
            }
            .case-analysis-content {
                display: none;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
            }
            @media (min-width: 3840px) {
                body { max-width: 80%; }
            }
            @media (max-width: 1200px) {
                .cases-row {
                    flex-direction: column;
                }
                .case-group {
                    width: 100%;
                    margin-bottom: 15px;
                }
            }
            @media (max-width: 768px) {
                body { max-width: 95%; padding: 10px; }
                .case-images { flex-direction: column; }
                .case-image-cell { width: 100%; margin-bottom: 10px; }
            }
        </style>
        """

        # JavaScript代码
        javascript = """
        <script>
            function openTab(evt, tabName) {
                let tabcontent = document.getElementsByClassName("tab-content");
                for (let i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                let tablinks = document.getElementsByClassName("tab-button");
                for (let i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
            
            function openCaseAnalysisTab(evt, tabName) {
                let tabcontent = document.getElementsByClassName("case-analysis-content");
                for (let i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                let tablinks = document.getElementsByClassName("case-analysis-tab-button");
                for (let i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
            
            document.addEventListener('DOMContentLoaded', function() {
                // 确保初始化时第一个tab是激活的
                let firstTab = document.querySelector('.tab-button');
                if (firstTab) {
                    firstTab.click();
                }
                
                // 确保每个数据集的第一个案例分析标签是激活的
                document.querySelectorAll('.case-analysis-tab-button').forEach(function(btn, index) {
                    if (index % 2 === 0) { // 每组的第一个按钮
                        btn.click();
                    }
                });
            });
        </script>
        """

        # 生成摘要表格
        summary_table = "<table>\n"
        summary_table += "<tr><th>数据集</th><th>缩放比例</th><th>样本数</th><th>平均 PSNR</th><th>平均 SSIM</th></tr>\n"

        overall_psnr = []
        overall_ssim = []

        for dataset_name, scale_results in all_results.items():
            for scale, results in scale_results.items():
                if results:
                    psnr_values = [r[1] for r in results]
                    ssim_values = [r[2] for r in results]
                    avg_psnr = np.mean(psnr_values)
                    avg_ssim = np.mean(ssim_values)
                    overall_psnr.extend(psnr_values)
                    overall_ssim.extend(ssim_values)

                    summary_table += f"<tr><td>{dataset_name}</td><td>x{scale}</td><td>{len(results)}</td>"
                    summary_table += (
                        f"<td>{avg_psnr:.4f}</td><td>{avg_ssim:.4f}</td></tr>\n"
                    )

        if overall_psnr:
            summary_table += (
                f'<tr style="font-weight: bold; background-color: #e6ffe6;">'
            )
            summary_table += f'<td colspan="2">总计</td><td>{len(overall_psnr)}</td>'
            summary_table += f"<td>{np.mean(overall_psnr):.4f}</td><td>{np.mean(overall_ssim):.4f}</td></tr>\n"

        summary_table += "</table>\n"

        # 生成HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>超分辨率评估报告</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            {style}
        </head>
        <body>
            <div class="header">
                <h1>超分辨率评估报告</h1>
                <p>多数据集、多尺度超分辨率结果评估与可视化分析</p>
            </div>
            <div class="section">
                <h2>总体摘要</h2>
                <div class="summary">
                    <div class="summary-card">
                        <h3>数据集数量</h3>
                        <div class="value">{len(all_results)}</div>
                    </div>
                    <div class="summary-card">
                        <h3>样本总数</h3>
                        <div class="value">{len(overall_psnr)}</div>
                    </div>
                    <div class="summary-card">
                        <h3>整体 PSNR</h3>
                        <div class="value">{np.mean(overall_psnr):.2f} dB</div>
                    </div>
                    <div class="summary-card">
                        <h3>整体 SSIM</h3>
                        <div class="value">{np.mean(overall_ssim):.4f}</div>
                    </div>
                </div>
                {summary_table}
            </div>
            <div class="section">
                <h2>数据集详细结果</h2>
                <div class="tabs">
        """

        # 添加数据集标签页按钮
        for i, dataset_name in enumerate(all_results.keys()):
            active_class = "active" if i == 0 else ""
            html_content += f'<button class="tab-button {active_class}" onclick="openTab(event, \'{dataset_name}\')">{dataset_name}</button>\n'

        html_content += "</div>\n"

        # 添加每个数据集的内容
        for i, (dataset_name, scale_results) in enumerate(all_results.items()):
            display_style = "block" if i == 0 else "none"
            html_content += f'<div id="{dataset_name}" class="tab-content" style="display: {display_style};">\n'

            for scale, results in scale_results.items():
                if results:
                    dataset_dir = os.path.join(root_dir, dataset_name)
                    html_content += f"<h3>{dataset_name} - x{scale} 缩放</h3>\n"
                    html_content += f'<div class="chart">\n'
                    html_content += f'<img src="charts/{dataset_name}_x{scale}_distribution.png" alt="{dataset_name} x{scale} 分布">\n'
                    html_content += "</div>\n"

                    # 添加最佳/最差案例分析选项卡
                    html_content += f"""
                    <div class="case-analysis-tabs">
                        <button class="case-analysis-tab-button active" onclick="openCaseAnalysisTab(event, 'best-{dataset_name}-{scale}')">最佳案例</button>
                        <button class="case-analysis-tab-button" onclick="openCaseAnalysisTab(event, 'worst-{dataset_name}-{scale}')">最差案例</button>
                    </div>
                    """

                    # 生成最佳案例HTML
                    best_cases_html = Visualizer.generate_cases_analysis_html(
                        dataset_dir,
                        results,
                        scale,
                        output_path,
                        "best",
                        patch_size,
                        interp_method,
                        num_cases,
                    )

                    # 生成最差案例HTML
                    worst_cases_html = Visualizer.generate_cases_analysis_html(
                        dataset_dir,
                        results,
                        scale,
                        output_path,
                        "worst",
                        patch_size,
                        interp_method,
                        num_cases,
                    )

                    # 添加最佳案例内容
                    html_content += f"""
                    <div id="best-{dataset_name}-{scale}" class="case-analysis-content" style="display: block;">
                        {best_cases_html}
                    </div>
                    """

                    # 添加最差案例内容
                    html_content += f"""
                    <div id="worst-{dataset_name}-{scale}" class="case-analysis-content">
                        {worst_cases_html}
                    </div>
                    """

            html_content += "</div>\n"

        html_content += f"""
            </div>
            {javascript}
        </body>
        </html>
        """

        # 创建输出目录并保存HTML报告
        os.makedirs(os.path.join(output_path, "charts"), exist_ok=True)
        with open(os.path.join(output_path, "report.html"), "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"HTML报告已保存到: {os.path.join(output_path, 'report.html')}")


# ===============================
# 4. 主程序入口
# ===============================
def main():
    parser = argparse.ArgumentParser(description="SR图像推理、评估和可视化")

    # 基本参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--root_dir", type=str, default="SR_Test", help="数据集根目录")
    parser.add_argument(
        "--scale", type=int, default=2, choices=[2, 3, 4], help="超分辨率比例因子"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="使用的设备")
    parser.add_argument(
        "--output_dir", type=str, default="SR_Results", help="结果输出目录"
    )

    # 新增的可配置参数
    parser.add_argument("--patch_size", type=int, default=256, help="分析用的区域大小")
    parser.add_argument(
        "--interp_method",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="用于比较的插值方法",
    )
    parser.add_argument(
        "--num_cases", type=int, default=12, help="每个数据集显示的最佳/最差案例数量"
    )
    parser.add_argument("--chart_dpi", type=int, default=100, help="保存图表的DPI")
    parser.add_argument("--chart_width", type=int, default=15, help="图表宽度(英寸)")
    parser.add_argument("--chart_height", type=int, default=9, help="图表高度(英寸)")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="并行处理的工作线程数，默认使用所有CPU核心",
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f"正在加载模型: {args.model_path}")
    from tools.models.efdn import EFDN
    from tools.models.lkfn import LKFN
    from tools.models.catanet import CATANet
    from tools.models.swinir import SwinIR

    # 选择模型 (可根据需要调整)
    model = LKFN()

    # model.load_state_dict(torch.load("checkpoints/epoch_1000.pth"))
    # 加载模型权重
    model = load_model(model, args.model_path)
    model = model.to(args.device)
    model.eval()

    # 获取可用的数据集
    available_datasets = ["Manga109", "Urban100", "BSD100", "Set5", "Set14"]
    datasets = [
        d for d in available_datasets if os.path.exists(os.path.join(args.root_dir, d))
    ]

    if not datasets:
        print(f"在 {args.root_dir} 中未找到有效的数据集")
        return

    # 处理每个数据集
    all_results = {}
    for dataset in datasets:
        dataset_path = os.path.join(args.root_dir, dataset)
        lr_path = os.path.join(dataset_path, f"x{args.scale}")
        pred_path = os.path.join(dataset_path, f"PRED_x{args.scale}")

        # 创建预测输出目录
        os.makedirs(pred_path, exist_ok=True)

        print(f"\n处理数据集: {dataset} (x{args.scale})")

        # 模型推理
        print(f"推理 {dataset} 的LR图像...")
        infer_from_model(model, lr_path, pred_path, device=args.device)

        # 评估结果
        print(f"评估 {dataset} 的推理结果...")
        results = MetricsCalculator.evaluate_sr_results(
            dataset_path, dataset, args.scale, args.num_workers
        )

        if results and len(results) > 0:
            if dataset not in all_results:
                all_results[dataset] = {}
            all_results[dataset][args.scale] = results

            # 计算和输出指标
            psnr_values = [r[1] for r in results]
            ssim_values = [r[2] for r in results]
            print(f"{dataset} (x{args.scale}) 结果:")
            print(f"  样本数: {len(results)}")
            print(f"  平均 PSNR: {np.mean(psnr_values):.4f} dB")
            print(f"  平均 SSIM: {np.mean(ssim_values):.4f}")
            print(f"  最高 PSNR: {np.max(psnr_values):.4f} dB")
            print(f"  最低 PSNR: {np.min(psnr_values):.4f} dB")

            # 创建可视化
            print(f"创建 {dataset} 的可视化...")
            dist_fig = Visualizer.plot_dataset_results(
                results,
                dataset,
                args.scale,
                figsize=(args.chart_width, args.chart_height),
                dpi=args.chart_dpi,
            )

            if dist_fig:
                chart_dir = os.path.join(args.output_dir, "charts")
                os.makedirs(chart_dir, exist_ok=True)
                dist_fig.savefig(
                    os.path.join(
                        chart_dir, f"{dataset}_x{args.scale}_distribution.png"
                    ),
                    dpi=args.chart_dpi,
                    bbox_inches="tight",
                )
                plt.close(dist_fig)

    # 创建HTML报告
    if all_results:
        print("\n创建汇总报告...")
        patch_size = (args.patch_size, args.patch_size)
        ReportGenerator.create_html_report(
            all_results,
            args.output_dir,
            args.root_dir,
            patch_size,
            args.interp_method,
            args.num_cases,
        )
        print(f"评估完成！结果已保存到 {args.output_dir}")
    else:
        print("没有结果可以显示")


if __name__ == "__main__":
    main()
