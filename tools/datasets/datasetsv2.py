import os
import random
import glob
import cv2
import numpy as np
import torch
import torch.utils.data as data
from configs.option import get_option
from tqdm import tqdm


def degrade_image(image, scale=4, output_shape=(256, 256)):
    """
    接受一张高分辨率(HR)图像，应用随机的退化操作生成对应的低分辨率(LR)图像。
    支持多种降采样比例、盲退化方式和数据增强，输入和输出格式可灵活选择。
    参数:
        image: 输入的HR图像。可以是PIL.Image对象、NumPy数组或torch.Tensor。
        scale: 降采样比例，例如2、4、8等。
        output_shape: 输出HR图像大小，格式为(H, W)。
    返回:
        (hr_image, lr_image): 元组，其中第一个是处理后的HR图像（裁剪或填充后为output_shape大小），
                              第二个是生成的对应LR图像，其尺寸为(output_shape除以scale)。
    """

    # 获取图像尺寸
    H, W, C = image.shape

    # 1. 裁剪或填充到指定尺寸
    if H < output_shape[0] or W < output_shape[1]:
        # 如果图像尺寸小于指定尺寸，则填充
        pad_h = max(0, output_shape[0] - H)
        pad_w = max(0, output_shape[1] - W)
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    elif H > output_shape[0] or W > output_shape[1]:
        # 如果图像尺寸大于指定尺寸，则随机裁剪
        y = random.randint(0, H - output_shape[0])
        x = random.randint(0, W - output_shape[1])
        image = image[y : y + output_shape[0], x : x + output_shape[1]]
    # 更新尺寸
    H, W = image.shape[0], image.shape[1]
    # 增强后的HR图像作为输出高分辨率图像（不含模糊/噪声等退化）
    hr_img = image.astype(np.float32)  # 提前转换为float32以减少类型转换
    lr_img = hr_img.copy()  # 使用copy避免修改原始图像

    # 随机选择要应用的退化效果
    apply_blur = random.random() < 0.66
    apply_noise = random.random() < 0.0
    apply_jpeg = random.random() < 0.0

    # 3. 盲退化策略：模糊（高斯模糊或运动模糊）
    if apply_blur:
        blur_type = random.random()
        if blur_type < 0.5:
            # 高斯模糊
            sigma = random.uniform(0.5, 3.0)
            lr_img = cv2.GaussianBlur(lr_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            # 运动模糊 - 预先计算常用核
            angle = random.choice([0, 45, 90, 135])
            k = random.choice([3, 5, 7, 9, 11, 13, 15])

            if angle == 0:
                kernel = np.zeros((1, k), np.float32)
                kernel[0, :] = 1.0 / k
            elif angle == 90:
                kernel = np.zeros((k, 1), np.float32)
                kernel[:, 0] = 1.0 / k
            elif angle == 45:
                kernel = np.eye(k, dtype=np.float32) / k
            else:  # angle == 135
                kernel = np.zeros((k, k), np.float32)
                kernel[np.arange(k), k - 1 - np.arange(k)] = 1.0 / k

            lr_img = cv2.filter2D(lr_img, -1, kernel)

    # 盲退化策略：噪声（高斯噪声或泊松噪声）
    if apply_noise:
        noise_type = random.random()
        if noise_type < 0.5:
            # 添加高斯噪声
            sigma_n = random.uniform(1, 10)
            noise = np.random.normal(0, sigma_n, lr_img.shape)
            lr_img += noise
        else:
            # 添加泊松噪声 - 直接在浮点数上操作
            lr_img = np.random.poisson(np.maximum(1, lr_img)).astype(np.float32)

    # 盲退化策略：颜色偏移与伪影（对RGB三个通道随机缩放）
    factors = np.array([random.uniform(0.9, 1.1) for _ in range(3)], dtype=np.float32)
    lr_img = lr_img * factors.reshape(1, 1, 3)

    # 裁剪到有效范围
    lr_img = np.clip(lr_img, 0, 255)

    # 盲退化策略：随机下采样（降分辨率）
    new_h, new_w = max(1, int(H // scale)), max(1, int(W // scale))
    interp = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    lr_img = cv2.resize(lr_img.astype(np.uint8), (new_w, new_h), interpolation=interp)

    # 盲退化策略：JPEG压缩造成的伪影
    if apply_jpeg:
        quality = random.randint(30, 95)
        # 避免不必要的颜色空间转换
        _, enc = cv2.imencode(".jpg", lr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        lr_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 确保输出是uint8类型
    hr_img = hr_img.astype(np.uint8)
    if lr_img.dtype != np.uint8:
        lr_img = lr_img.astype(np.uint8)

    return hr_img, lr_img


class ImageDataset(data.Dataset):
    """
    图像数据集类。

    Args:
        phase (str): 数据集阶段，'train' 或 'valid'。
        opt (object): 配置选项。
        train_transform (callable, optional): 训练数据转换函数。
        valid_transform (callable, optional): 验证数据转换函数。
    """

    def __init__(self, phase, opt):
        self.phase = phase
        self.opt = opt
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

        self.loaded_images = {}  # 存储已加载的图像
        self.load_images_in_parallel()

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

        if image_path in self.loaded_images:  # 从缓存中获取图像
            hr_image = self.loaded_images[image_path]
        else:
            hr_image = self._load_image(image_path)

        if self.phase == "train":
            hr_image, lr_image = degrade_image(hr_image, scale=self.upscaling_factor)
        else:
            h, w, _ = hr_image.shape
            target_size = 1536

            # Padding to reach target size
            if h < target_size or w < target_size:
                pad_h = max(0, target_size - h)
                pad_w = max(0, target_size - w)
                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left
                hr_image = cv2.copyMakeBorder(
                    hr_image, top, bottom, left, right, cv2.BORDER_REFLECT_101
                )

            if hr_image.shape[0] > target_size or hr_image.shape[1] > target_size:
                h, w = hr_image.shape[:2]
                start_h = (h - target_size) // 2
                start_w = (w - target_size) // 2
                hr_image = hr_image[
                    start_h : start_h + target_size, start_w : start_w + target_size
                ]
            lr_image = cv2.resize(
                hr_image,
                (
                    hr_image.shape[1] // self.upscaling_factor,
                    hr_image.shape[0] // self.upscaling_factor,
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        lr_image = torch.from_numpy(lr_image).permute(2, 0, 1).float() / 127.5 - 1
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1).float() / 127.5 - 1

        if lr_image is not None and hr_image is not None:
            return {"lr_image": lr_image.float(), "hr_image": hr_image.float()}
        else:
            return None

    def __len__(self):
        return len(self.image_list)

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_images_in_parallel(self):
        """使用多线程预加载所有图像。"""
        import concurrent.futures

        print(f"开始使用多线程预加载 {len(self.image_list)} 张图像...")

        # 创建进度条
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # 提交所有任务
            future_to_path = {
                executor.submit(self._load_image, path): path
                for path in self.image_list
            }

            # 使用tqdm创建进度条
            with tqdm(
                total=len(self.image_list), desc="加载图像", mininterval=0.5
            ) as pbar:
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        self.loaded_images[path] = future.result()
                    except Exception as e:
                        print(f"加载图像 {path} 时出错: {e}")
                    pbar.update(1)

        print("图像预加载完成。")


def get_dataloader(opt):
    """
    获取数据加载器。

    Args:
        opt (object): 配置选项。

    Returns:
        tuple: 训练数据加载器和验证数据加载器。
    """
    train_dataset = ImageDataset(phase="train", opt=opt)
    valid_dataset = ImageDataset(phase="valid", opt=opt)
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
        batch_size=4,  # 验证集批量大小通常可以设置大一些
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    opt = get_option()
    train_dataloader, valid_dataloader = get_dataloader(opt)

    for i, batch in enumerate(train_dataloader):
        print(batch["lr_image"].shape, batch["hr_image"].shape)
        break
