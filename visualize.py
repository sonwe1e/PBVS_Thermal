import torch
import matplotlib.pyplot as plt
from typing import Union, Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader
import numpy as np
from configs.option import get_option
from tools.datasets.datasets import *

torch.set_float32_matmul_precision("high")


class ImageVisualizer:
    def __init__(
        self,
        opt: Optional[Any] = None,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
    ):
        """
        图像可视化工具类（适用于超分任务）

        Args:
            mean: 图像归一化均值
            std: 图像归一化标准差
        """
        self.opt = opt
        self.mean = torch.tensor(mean).reshape(3, 1, 1)
        self.std = torch.tensor(std).reshape(3, 1, 1)

    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        反归一化图像

        Args:
            image: 输入图像张量

        Returns:
            反归一化后的图像张量
        """
        return image + 0.2
        # return image * self.std + self.mean

    def plot_sr_grid(
        self,
        images: list,  # 修改为列表类型
        titles: list,
        nrow: int = 4,
        title: str = None,
        save_path: str = "./visualization/SR_comparison.png",
    ):
        """
        绘制超分任务图像网格（支持不同尺寸图像显示）

        Args:
            images: 图像张量列表（需包含LR和HR图像）
            titles: 每个子图的标题列表
            nrow: 每行显示的图像数量（推荐设置为配对数量）
            title: 整个图表的标题
            save_path: 图像保存路径
        """
        n = len(images)
        plt.rcParams.update({"font.size": 10})

        # 创建网格布局（每个样本占一行，LR左，HR右）
        fig, axes = plt.subplots(
            n // nrow,
            nrow,
            figsize=(nrow * 3, (n // nrow) * 3),
            gridspec_kw={"wspace": 0.05, "hspace": 0.2},
        )

        # 处理单样本情况
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)

        # 绘制每个图像
        for idx, ax in enumerate(axes.flat):
            if idx < len(images):  # 使用列表长度代替n
                # 处理单个图像
                img = self.denormalize(images[idx]).permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                ax.imshow(img, interpolation="nearest")  # 添加插值方法
                ax.set_title(titles[idx], fontsize=12)
                ax.axis("off")
            else:
                ax.axis("off")

        # 添加总标题并保存
        if title:
            plt.suptitle(title, y=0.98, fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return fig


def get_batch_data(
    batch: Union[Tuple, Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从batch中提取LR和HR图像

    Args:
        batch: 数据批次

    Returns:
        (lr_images, hr_images) 元组
    """
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]  # 假设返回顺序是（LR, HR）
    elif isinstance(batch, dict):
        return batch["lr_image"], batch["hr_image"]
    raise ValueError("Unsupported batch format. Expected tuple, list or dict.")


def visualize_sr_datasets(
    opt: Any,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    num_pairs: int = 8,
) -> None:
    """
    超分数据集可视化

    Args:
        opt: 配置选项
        train_dataloader: 训练数据加载器
        valid_dataloader: 验证数据加载器
        num_pairs: 要可视化的图像对数
    """
    visualizer = ImageVisualizer(opt)

    # 获取训练集数据
    train_lr, train_hr = get_batch_data(next(iter(train_dataloader)))
    train_lr = train_lr[:num_pairs]
    train_hr = train_hr[:num_pairs]

    # 获取验证集数据
    valid_lr, valid_hr = get_batch_data(next(iter(valid_dataloader)))
    valid_lr = valid_lr[:num_pairs]
    valid_hr = valid_hr[:num_pairs]

    # 合并所有图像和生成标题
    all_images = []
    titles = []

    # 添加训练集样本
    for i in range(num_pairs):
        all_images.extend([train_lr[i], train_hr[i]])
        titles.extend([f"Train LR", f"Train HR"])

    # 添加验证集样本
    for i in range(num_pairs):
        all_images.extend([valid_lr[i], valid_hr[i]])
        titles.extend([f"Valid LR", f"Valid HR"])

    # 转换为张量
    combined_images = all_images

    # 绘制可视化结果
    visualizer.plot_sr_grid(
        images=combined_images,
        titles=titles,
        nrow=10,  # 每行显示一对（LR和HR）
        title="Super Resolution Data Comparison",
        save_path="./visualization/SR_comparison.png",
    )


def main():
    """主函数"""
    opt = get_option()
    train_dataloader, valid_dataloader = get_dataloader(opt)

    print("Starting SR data visualization...")
    visualize_sr_datasets(opt, train_dataloader, valid_dataloader)
    print("Visualization saved to ./visualization/SR_comparison.png")


if __name__ == "__main__":
    main()
