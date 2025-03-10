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


def degrade_image(image, scale=2, output_shape=(448, 800)):
    """
    接受一张高分辨率(HR)图像，应用随机的退化操作生成对应的低分辨率(LR)图像。
    支持多种降采样比例、盲退化方式和数据增强，输入和输出格式可灵活选择。
    参数:
        image: 输入的HR图像。可以是PIL.Image对象、NumPy数组或torch.Tensor。
        scale: 降采样比例，例如2、4、8等。
        output_shape: 输出HR图像大小，格式为(H, W)。默认为(448, 800)。
    返回:
        (hr_image, lr_image): 元组，其中第一个是处理后的HR图像（裁剪或填充后为output_shape大小），
                              第二个是生成的对应LR图像，其尺寸为(output_shape除以scale)。
    """
    if isinstance(image, np.ndarray):
        img_np = image.copy()
        if img_np.ndim == 2:
            # 灰度图扩展为三通道
            img_np = np.stack([img_np] * 3, axis=2)
        elif img_np.ndim == 3:
            img_np = img_np[:, :, ::-1]  # BGR转RGB
    else:
        img_np = np.array(image.convert("RGB"), dtype=np.uint8)

    # 获取图像尺寸
    H, W, C = img_np.shape

    # 2. 数据增强：随机翻转
    if random.random() < 0.5:
        img_np = np.flip(img_np, axis=1)  # 水平翻转
    if random.random() < 0.5:
        img_np = np.flip(img_np, axis=0)  # 垂直翻转
    # 随机旋转90度的倍数
    rot_k = random.choice([0, 1, 2, 3])  # 0不旋转，1=90°, 2=180°, 3=270°
    if rot_k != 0:
        img_np = np.rot90(img_np, k=rot_k)
    # 更新尺寸
    H, W = img_np.shape[0], img_np.shape[1]

    # 随机裁剪并填充（保持尺寸不变）的数据增强
    if random.random() < 0.5:
        random_h = random.random() * 0.3
        random_w = random.random() * 0.3
        max_crop_h = max(1, int(random_h * H))
        max_crop_w = max(1, int(random_w * W))
        dx = random.randint(-max_crop_w, max_crop_w)  # 水平位移（正值表示向右移动）
        dy = random.randint(-max_crop_h, max_crop_h)  # 垂直位移（正值表示向下移动）
        # 计算裁剪区域
        left_crop = max(-dx, 0)  # 若dx为负，裁剪左侧 |dx| 列
        right_crop = max(dx, 0)  # 若dx为正，裁剪右侧 dx 列
        top_crop = max(-dy, 0)  # dy为负，裁剪上方 |dy| 行
        bottom_crop = max(dy, 0)  # dy为正，裁剪下方 dy 行
        rem_h = H - top_crop - bottom_crop  # 剩余高度
        rem_w = W - left_crop - right_crop  # 剩余宽度
        if rem_h > 0 and rem_w > 0:
            # 创建新图像并将裁剪后的原图放置到新图像中
            new_img = np.zeros((H, W, C), dtype=img_np.dtype)
            new_img[top_crop : top_crop + rem_h, left_crop : left_crop + rem_w] = (
                img_np[top_crop : top_crop + rem_h, left_crop : left_crop + rem_w]
            )
            img_np = new_img
            # 图像尺寸H, W保持不变

    # 增强后的HR图像作为输出高分辨率图像（不含模糊/噪声等退化）
    hr_img = img_np.copy()

    # 复制一份用于生成退化的LR图像
    lr_img = hr_img.copy()

    # 3. 盲退化策略：模糊（高斯模糊或运动模糊）
    blur_choice = random.random()
    if blur_choice < 0.33:
        # 高斯模糊
        sigma = random.uniform(0.5, 3.0)  # 随机选择模糊程度
        lr_img = cv2.GaussianBlur(lr_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    elif blur_choice < 0.66:
        # 运动模糊
        angle = random.choice([0, 45, 90, 135])  # 随机方向
        k = random.choice([3, 5, 7, 9, 11, 13, 15])  # 随机核长度（取奇数）
        if angle == 0:
            # 水平运动模糊
            kernel = np.zeros((1, k), np.float32)
            kernel[0, :] = 1.0 / k
        elif angle == 90:
            # 垂直运动模糊
            kernel = np.zeros((k, 1), np.float32)
            kernel[:, 0] = 1.0 / k
        elif angle == 45:
            # 45°对角线运动模糊
            kernel = np.zeros((k, k), np.float32)
            np.fill_diagonal(kernel, 1)
            kernel /= k
        else:
            # 135°对角线运动模糊
            kernel = np.zeros((k, k), np.float32)
            for i in range(k):
                kernel[i, k - 1 - i] = 1.0
            kernel /= k
        # 注意：OpenCV默认使用BGR，所以先转换
        lr_img_bgr = cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR)
        lr_img_bgr = cv2.filter2D(lr_img_bgr, -1, kernel)
        lr_img = cv2.cvtColor(lr_img_bgr, cv2.COLOR_BGR2RGB)
    # 如果 blur_choice >= 0.66，则不进行模糊

    # 盲退化策略：噪声（高斯噪声或泊松噪声）
    noise_choice = random.random()
    if noise_choice < 0.33:
        # 添加高斯噪声
        lr_float = lr_img.astype(np.float32)
        sigma_n = random.uniform(1, 10)
        noise = np.random.normal(0, sigma_n, lr_float.shape)
        lr_float += noise
        lr_img = np.clip(lr_float, 0, 255).astype(np.uint8)
    elif noise_choice < 0.66:
        # 添加泊松噪声
        lr_float = lr_img.astype(np.float32)
        vals = np.random.poisson(lr_float)
        lr_img = np.clip(vals, 0, 255).astype(np.uint8)
    # 若 noise_choice >= 0.66，则不加噪声

    # 盲退化策略：颜色偏移与伪影（对RGB三个通道随机缩放）
    lr_float = lr_img.astype(np.float32)
    factors = [random.uniform(0.9, 1.1) for _ in range(3)]
    for c in range(3):
        lr_float[:, :, c] *= factors[c]
    lr_img = np.clip(lr_float, 0, 255).astype(np.uint8)

    # 盲退化策略：随机下采样（降分辨率）
    # 注意：cv2.resize 的参数顺序为 (width, height)
    new_h = max(1, int(H // scale))
    new_w = max(1, int(W // scale))
    interp = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    lr_img = cv2.resize(lr_img, (new_w, new_h), interpolation=interp)

    # 盲退化策略：JPEG压缩造成的伪影
    if random.random() < 0.5:
        quality = random.randint(30, 95)
        _, enc = cv2.imencode(
            ".jpg",
            cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        lr_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

    # -------------------------------
    # 5. 最终裁剪与填充：保证 HR 图像为 output_shape，且裁剪起点对 scale 取整
    desired_h, desired_w = output_shape
    # 如果 HR 图像尺寸不足，则进行填充（使用黑色填充）
    pad_h = max(0, desired_h - hr_img.shape[0])
    pad_w = max(0, desired_w - hr_img.shape[1])
    if pad_h > 0 or pad_w > 0:
        hr_img = cv2.copyMakeBorder(
            hr_img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=0
        )
        # 同步对 lr_img 进行填充前的更新：由于 lr_img 已经下采样，
        # 如果 HR 经过填充后尺寸变化，LR 需要重新计算尺寸（近似用当前插值方式）
        new_lr_h = hr_img.shape[0] // scale
        new_lr_w = hr_img.shape[1] // scale
        lr_img = cv2.resize(lr_img, (new_lr_w, new_lr_h), interpolation=interp)

    H, W, _ = hr_img.shape
    # 选择随机裁剪位置，确保裁剪区域大小为 output_shape
    max_top = H - desired_h
    max_left = W - desired_w
    top = random.randint(0, max_top)
    left = random.randint(0, max_left)
    # 调整裁剪起点为 scale 的倍数
    top = top - (top % scale)
    left = left - (left % scale)

    hr_img_cropped = hr_img[top : top + desired_h, left : left + desired_w]
    # 对应 LR 图像裁剪：由于 lr_img 为 HR 下采样后的图像，
    # 则裁剪起点为 (top//scale, left//scale)，裁剪尺寸为 (desired_h//scale, desired_w//scale)
    lr_top = top // scale
    lr_left = left // scale
    desired_h_lr = desired_h // scale
    desired_w_lr = desired_w // scale
    lr_img_cropped = lr_img[
        lr_top : lr_top + desired_h_lr, lr_left : lr_left + desired_w_lr
    ]
    # -------------------------------

    # 最终返回裁剪后的图像
    out_hr, out_lr = hr_img_cropped.astype(np.uint8), lr_img_cropped.astype(np.uint8)
    return out_hr, out_lr


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
        self.hr_path = os.path.join(self.data_path, phase)

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
                image_extensions = ["*.png", "*.bmp", "*.jpg", "*.jpeg"]
                for ext in image_extensions:
                    self.dynamic_hr_image_paths.extend(
                        glob.glob(os.path.join(extra_data_path, ext))
                    )
                    print(
                        f"加载额外数据: {extra_data_path} 中的图像文件: {len(self.dynamic_hr_image_paths)}"
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
            # if image_path in self.dynamic_hr_image_paths:
            hr_image = self._random_crop(hr_image, 448, 512)

            # 使用 bicubic 下采样生成 LR 图像
            lr_image = cv2.resize(
                hr_image,
                (
                    hr_image.shape[1] // self.upscaling_factor,
                    hr_image.shape[0] // self.upscaling_factor,
                ),
                interpolation=cv2.INTER_CUBIC,
            )

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
        return len(self.image_list)

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

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

        if random.random() < 0.3:
            sigma = random.uniform(0, 5)  # 控制噪声强度
            noise = np.random.normal(0, sigma, lr_image.shape).astype(np.float32)
            lr_image = cv2.add(lr_image.astype(np.float32), noise)
            lr_image = np.clip(lr_image, 0, 255).astype(np.uint8)

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
        drop_last=True,
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
