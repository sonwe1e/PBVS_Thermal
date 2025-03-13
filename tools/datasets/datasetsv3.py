import os
import random
import glob
import cv2
import torch
import torch.utils.data as data
import queue as Queue
import threading
from configs.option import get_option
from tqdm import tqdm
import numpy as np


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Reference: https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(data.DataLoader):
    """Prefetch version of dataloader.

    Reference: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher:
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher:
    """CUDA prefetcher.

    Reference: https://github.com/NVIDIA/apex/issues/304#

    It may consume more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device(
            f"cuda:{opt.devices[0]}" if len(opt.devices) != 0 else "cpu"
        )
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(
                        device=self.device, non_blocking=True
                    )

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()


# 新增：可枚举的预取器迭代器类
class PrefetcherIterator:
    """可迭代的预取器包装类，使预取器可以在for循环中使用enumerate。

    Args:
        prefetcher: CPU或CUDA预取器。
    """

    def __init__(self, prefetcher, length=None):
        self.prefetcher = prefetcher
        self.length = length
        self.iterator = None

    def __iter__(self):
        self.prefetcher.reset()
        batch = self.prefetcher.next()

        iterator = range(self.length) if self.length is not None else iter(int, 1)

        for i in iterator:
            if batch is None:
                break
            yield batch
            batch = self.prefetcher.next()

    def __len__(self):
        return self.length if self.length is not None else 0


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
        if self.phase == "train" and hasattr(opt, "extra_data"):
            self._load_extra_data()

        self.image_list = self.hr_image_paths

        # 加载数据到内存中
        self.loaded_images = {}  # 存储已加载的图像
        self.load_images_in_parallel()

        self.image_list = self.image_list + self.dynamic_hr_image_paths

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
            h, w, _ = hr_image.shape
            crop_size = 256
            start_h = random.randint(0, h - crop_size)
            start_w = random.randint(0, w - crop_size)
            hr_image = hr_image[
                start_h : start_h + crop_size, start_w : start_w + crop_size
            ]

            # 下采样得到 lr_image
            lr_image = cv2.resize(
                hr_image,
                (
                    hr_image.shape[1] // self.upscaling_factor,
                    hr_image.shape[0] // self.upscaling_factor,
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            h, w, _ = hr_image.shape
            target_h = 1536
            target_w = 2048

            # Padding to reach target size
            if h < target_h or w < target_w:
                pad_h = max(0, target_h - h)
                pad_w = max(0, target_w - w)
                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left
                hr_image = cv2.copyMakeBorder(
                    hr_image, top, bottom, left, right, cv2.BORDER_CONSTANT
                )

            if hr_image.shape[0] > target_h or hr_image.shape[1] > target_w:
                h, w = hr_image.shape[:2]
                start_h = (h - target_h) // 2
                start_w = (w - target_w) // 2
                hr_image = hr_image[
                    start_h : start_h + target_h, start_w : start_w + target_w
                ]
            lr_image = cv2.resize(
                hr_image,
                (
                    hr_image.shape[1] // self.upscaling_factor,
                    hr_image.shape[0] // self.upscaling_factor,
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        lr_image = torch.from_numpy(lr_image).permute(2, 0, 1).float() / 255.0
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1).float() / 255.0

        if lr_image is not None and hr_image is not None:
            return {"lr_image": lr_image.float(), "hr_image": hr_image.float()}
        else:
            return None

    def __len__(self):
        return len(self.image_list)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_images_in_parallel(self):
        """使用多线程预加载所有图像。"""
        import concurrent.futures

        print(f"开始使用多线程预加载 {len(self.image_list)} 张图像...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
            future_to_path = {
                executor.submit(self._load_image, path): path
                for path in self.image_list
            }

            # tqdm is excellent for progress bars, keep using it!
            with tqdm(
                total=len(self.image_list),
                desc="Loading Images",
                mininterval=0.5,
                unit="img",
            ) as pbar:
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        # 3. Store Directly: You're already doing this correctly!
                        self.loaded_images[path] = future.result()
                    except Exception as e:
                        # 4.  More Specific Error Handling (Optional but Good Practice)
                        print(f"Error loading image {path}: {e}")
                        # Consider logging the error, or adding the path to a list of failed images.
                    finally:
                        pbar.update(1)


def get_dataloader(opt):
    """
    获取数据加载器。

    Args:
        opt (object): 配置选项。

    Returns:
        tuple: 训练和验证数据的预取迭代器和原始加载器。
    """
    train_dataset = ImageDataset(phase="train", opt=opt)
    valid_dataset = ImageDataset(phase="valid", opt=opt)

    # 使用PrefetchDataLoader替代普通DataLoader
    train_dataloader = PrefetchDataLoader(
        num_prefetch_queue=4,
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_dataloader = PrefetchDataLoader(
        num_prefetch_queue=4,
        dataset=valid_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    # 根据GPU可用性创建对应的预取器
    if len(opt.devices) > 0 and torch.cuda.is_available():
        train_prefetcher = CUDAPrefetcher(train_dataloader, opt)
        valid_prefetcher = CUDAPrefetcher(valid_dataloader, opt)
    else:
        train_prefetcher = CPUPrefetcher(train_dataloader)
        valid_prefetcher = CPUPrefetcher(valid_dataloader)

    # 创建可迭代的预取器
    train_iterator = PrefetcherIterator(
        train_prefetcher,
        length=len(train_dataset) // opt.batch_size,
    )

    valid_iterator = PrefetcherIterator(
        valid_prefetcher,
        length=len(valid_dataset) // 10,
    )

    return train_iterator, valid_iterator, train_dataloader, valid_dataloader


if __name__ == "__main__":
    opt = get_option()

    # 获取数据加载器和预取迭代器
    train_iterator, valid_iterator, train_dataloader, valid_dataloader = get_dataloader(
        opt
    )

    # 使用可枚举的预取迭代器
    print("使用可枚举的预取迭代器：")
    for i, batch in enumerate(train_iterator):
        print(f"Batch {i}:", batch["lr_image"].shape, batch["hr_image"].shape)
        print(batch["lr_image"].max(), batch["lr_image"].min())
        if i >= 2:  # 只演示前几个批次
            break

    # 也可以直接使用原始dataloader
    print("\n使用原始DataLoader：")
    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i}:", batch["lr_image"].shape, batch["hr_image"].shape)
        print(batch["lr_image"].max(), batch["lr_image"].min())
        if i >= 2:  # 只演示前几个批次
            break
