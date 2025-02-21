import os
import random
import glob
import cv2
import numpy as np
import torch
import torch.utils.data as data
from concurrent.futures import ThreadPoolExecutor
from tools.datasets.augments import train_transform, valid_transform
from configs.option import get_option


class ImageDataset(data.Dataset):
    """
    图像数据集类。

    Args:
        phase (str): 数据集阶段，'train' 或 'valid'。
        opt (object): 配置选项。
        train_transform (callable, optional): 训练数据转换函数。
        valid_transform (callable, optional): 验证数据转换函数。
    """

    def __init__(self, phase, opt, train_transform=None, valid_transform=None):
        self.phase = phase
        self.opt = opt
        self.transform = train_transform if phase == "train" else valid_transform
        self.upscaling_factor = opt.upscaling_factor

        # 数据路径
        self.data_path = opt.data_path
        self.hr_path = os.path.join(self.data_path, phase, "GT")

        # 高分辨率图像路径列表
        self.hr_image_paths = [
            os.path.join(self.hr_path, name) for name in os.listdir(self.hr_path)
        ]
        self.dynamic_hr_image_paths = []

        # 加载额外的训练数据 (仅训练阶段)
        if self.phase == "train" and hasattr(opt, "extra_data"):
            self._load_extra_data()
        # 合并图像路径列表
        self.image_list = self.hr_image_paths + self.dynamic_hr_image_paths

    def _load_extra_data(self):
        """加载额外的训练数据。"""
        extra_data_paths = self.opt.extra_data
        for extra_data_path in extra_data_paths:
            if os.path.isdir(extra_data_path):
                self.dynamic_hr_image_paths.extend(
                    glob.glob(os.path.join(extra_data_path, "*.png"))
                )
            else:
                print(
                    f"警告: opt.extra_data 路径 '{extra_data_path}' 不是有效目录。将不会加载此路径的额外数据。"
                )

    def __getitem__(self, index):
        """
        获取单个数据样本。

        Args:
            index (int): 样本索引。

        Returns:
            dict: 包含低分辨率图像和高分辨率图像的字典，如果图像加载失败则返回None。
        """
        image_path = self.image_list[index]
        hr_image = self._load_image(image_path)
        lr_image = None

        if hr_image is not None:
            if image_path in self.dynamic_hr_image_paths:
                hr_image = self._random_crop(hr_image, 448, 640)

            # 使用 bicubic 下采样生成 LR 图像
            lr_image = cv2.resize(
                hr_image,
                (
                    hr_image.shape[1] // self.upscaling_factor,
                    hr_image.shape[0] // self.upscaling_factor,
                ),
                interpolation=cv2.INTER_CUBIC,
            )
            # 示例：模糊+噪声+压缩
            # noisy = hr_image + np.random.normal(0, 10**0.5, hr_image.shape)
            # cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1).astype(
            #     np.uint8
            # )
            # noisy = cv2.resize(
            #     noisy,
            #     fx=1 / self.upscaling_factor,
            #     fy=1 / self.upscaling_factor,
            #     dsize=None,
            #     interpolation=cv2.INTER_CUBIC,
            # )
            # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            # _, encoded_image = cv2.imencode(".jpg", noisy, encode_param)
            # lr_image = cv2.imdecode(encoded_image, 0)
            # if lr_image.ndim == 2:
            #     lr_image = np.stack([lr_image] * 3, axis=-1)

        if self.phase == "train" and hr_image is not None:
            lr_image, hr_image = self._augment(lr_image, hr_image)  # 数据增强
        if self.transform is not None and lr_image is not None and hr_image is not None:
            lr_image = self.transform(image=lr_image)["image"] / 127.5 - 1
            hr_image = self.transform(image=hr_image)["image"] / 127.5 - 1

        if lr_image is not None and hr_image is not None:
            return {"lr_image": lr_image.float(), "hr_image": hr_image.float()}
        else:
            return None  # 处理图像加载失败的情况

    def __len__(self):
        """
        获取数据集长度。

        Returns:
            int: 数据集长度。
        """
        return len(self.image_list)

    def _load_image(self, path):
        """
        加载图像，并根据文件类型进行处理。

        Args:
            path (str): 图像路径。

        Returns:
            numpy.ndarray: 加载并处理后的图像，形状为 (H, W, 3)，RGB 格式。
                        如果加载失败，则返回 None。
        """
        try:
            if path.lower().endswith(".bmp"):
                img = cv2.imread(path)
                if img is None:
                    raise Exception(f"无法读取 BMP 图像: {path}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            elif path.lower().endswith(".png"):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 读取所有通道
                if img is None:
                    raise Exception(f"无法读取 PNG 图像: {path}")
                if img.ndim == 2:  # 灰度图
                    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 3:  # RGB 图
                    # chosen_channel = random.randint(0, 2)
                    # return np.stack([img[:, :, chosen_channel]] * 3, axis=-1)
                    return np.stack(
                        [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)] * 3, axis=-1
                    )
                else:
                    raise Exception(f"图像通道数异常: {path}, 通道数: {img.shape[2]}")

            else:  # 尝试作为普通 BGR 图像读取
                img = cv2.imread(path)
                if img is None:
                    raise Exception(f"无法读取图像: {path}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"加载图像时出错: {path}: {e}")
            return None  # 返回 None 表示加载失败

    def _random_crop(self, img, min_h, min_w):
        """随机裁剪图像，处理图像尺寸小于目标尺寸的情况"""
        h, w = img.shape[:2]

        # 如果 h 或 w 小于 min_h 或 min_w，考虑旋转或填充
        if h < min_h or w < min_w:
            # 填充图像
            top_pad = max(0, min_h - h)
            bottom_pad = max(0, min_h - h)
            left_pad = max(0, min_w - w)
            right_pad = max(0, min_w - w)

            # 使用零填充（黑色填充），可以调整为其他颜色
            img = cv2.copyMakeBorder(
                img,
                top_pad,
                bottom_pad,
                left_pad,
                right_pad,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

            # 更新 h, w 为填充后的新尺寸
            h, w = img.shape[:2]

        # 现在确保 h 和 w 都大于等于 min_h 和 min_w，进行裁剪
        top = random.randint(0, h - min_h)
        left = random.randint(0, w - min_w)

        return img[top : top + min_h, left : left + min_w]

    def _augment(self, lr_image, hr_image):
        """改进后的数据增强方法，包含多种几何变换与色彩抖动"""

        # 随机旋转（-30度到30度）
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            h, w = lr_image.shape[:2]
            center_lr = (w // 2, h // 2)
            M_lr = cv2.getRotationMatrix2D(center_lr, angle, 1.0)
            lr_image = cv2.warpAffine(
                lr_image,
                M_lr,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            hr_h, hr_w = hr_image.shape[:2]
            center_hr = (hr_w // 2, hr_h // 2)
            M_hr = cv2.getRotationMatrix2D(center_hr, angle, 1.0)
            hr_image = cv2.warpAffine(
                hr_image,
                M_hr,
                (hr_w, hr_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        # 随机平移（LR平移±10像素，HR对应±80像素）
        if random.random() < 0.5:
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)

            h, w = lr_image.shape[:2]
            M_lr = np.float32([[1, 0, dx], [0, 1, dy]])
            lr_image = cv2.warpAffine(
                lr_image, M_lr, (w, h), borderMode=cv2.BORDER_REFLECT_101
            )

            hr_h, hr_w = hr_image.shape[:2]
            M_hr = np.float32([[1, 0, dx * 8], [0, 1, dy * 8]])
            hr_image = cv2.warpAffine(
                hr_image, M_hr, (hr_w, hr_h), borderMode=cv2.BORDER_REFLECT_101
            )

        # 色彩抖动（同步调整LR/HR）
        if random.random() < 0.5:
            # 生成随机色彩参数
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)

            # 应用相同的色彩变换到LR和HR
            lr_image = self._apply_color_jitter(
                lr_image, brightness, contrast, saturation
            )
            hr_image = self._apply_color_jitter(
                hr_image, brightness, contrast, saturation
            )

        # 添加高斯噪声到LR
        if random.random() < 0.3:
            sigma = random.uniform(0, 5)  # 控制噪声强度
            noise = np.random.normal(0, sigma, lr_image.shape).astype(np.float32)
            lr_image = cv2.add(lr_image.astype(np.float32), noise)
            lr_image = np.clip(lr_image, 0, 255).astype(np.uint8)

        # 随机翻转（保持原有逻辑）
        if random.random() > 0.5:
            lr_image = cv2.flip(lr_image, 1)
            hr_image = cv2.flip(hr_image, 1)
        if random.random() > 0.5:
            lr_image = cv2.flip(lr_image, 0)
            hr_image = cv2.flip(hr_image, 0)

        return lr_image, hr_image

    def _apply_color_jitter(self, image, brightness, contrast, saturation):
        """应用色彩抖动辅助函数"""
        # 亮度调整
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

        # 对比度调整
        mean = cv2.mean(image)[0]
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=(1 - contrast) * mean)

        # 饱和度调整（仅限彩色图像）
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[..., 1] = hsv[..., 1] * saturation
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return image

    # def load_images_in_parallel(self):
    #     """
    #     并行加载图像路径（不直接加载图像以节省内存）。
    #     此函数目前未使用，但保留以供将来可能的优化使用。
    #     """

    #     with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
    #         # 并行构建预生成图像的路径,  现在 hr 路径
    #         # futures_lr = [
    #         #     executor.submit(os.path.join, self.lr_path, name)
    #         #     for name in self.image_names
    #         # ]
    #         futures_hr = [
    #             executor.submit(os.path.join, self.hr_path, name)
    #             for name in self.image_names
    #         ]

    #         # 如果有额外的训练数据，并行构建其路径
    #         if (
    #             self.phase == "train"
    #             and hasattr(self.opt, "extra_data")
    #             and self.opt.extra_data
    #             and os.path.isdir(self.opt.extra_data)
    #         ):
    #             extra_data_path = self.opt.extra_data
    #             futures_extra = [
    #                 executor.submit(os.path.join, extra_data_path, name)
    #                 for name in os.listdir(extra_data_path)
    #                 if name.endswith(".png")  # 假设额外路径中的所有 png 文件都是图像
    #             ]
    #             self.dynamic_hr_image_paths = [
    #                 future.result() for future in futures_extra
    #             ]

    #         # 获取预生成图像的路径
    #         # self.lr_image_paths = [future.result() for future in futures_lr] # 现在是空的
    #         self.hr_image_paths = [future.result() for future in futures_hr]

    #     # 合并图像路径列表
    #     self.image_list = self.hr_image_paths + self.dynamic_hr_image_paths
    #     self.hr_image_list = self.hr_image_paths + self.dynamic_hr_image_paths


def get_dataloader(opt):
    """
    获取数据加载器。

    Args:
        opt (object): 配置选项。

    Returns:
        tuple: 训练数据加载器和验证数据加载器。
    """
    train_dataset = ImageDataset(
        phase="train",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )
    valid_dataset = ImageDataset(
        phase="valid",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=40,  # 验证集批量大小通常可以设置大一些
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    opt = get_option()
    train_dataloader, valid_dataloader = get_dataloader(opt)

    for i, batch in enumerate(train_dataloader):
        print(
            batch["lr_image"].shape, batch["hr_image"].shape
        )  # 打印 lr 和 hr 图像的形状
        break
