import os
import random
import cv2
import torch
import torch.utils.data as data
import queue as Queue
import threading
from configs.option import get_option
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple


class PrefetchGenerator(threading.Thread):
    """A thread-based generator for prefetching data in the background.

    Inspired by: https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: The source generator to prefetch from.
        max_queue_size (int): Maximum size of the prefetch queue.
    """

    def __init__(self, generator, max_queue_size):
        super().__init__(daemon=True)
        self.queue = Queue.Queue(max_queue_size)
        self.generator = generator
        self.start()

    def run(self):
        """Fills the queue with items from the generator."""
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)  # Sentinel to indicate end

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    def __iter__(self):
        return self


class PrefetchDataLoader(data.DataLoader):
    """A DataLoader enhanced with prefetching capabilities.

    Adapted from: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    Args:
        max_queue_size (int): Maximum size of the prefetch queue.
        **kwargs: Arguments passed to torch.utils.data.DataLoader.
    """

    def __init__(self, max_queue_size, **kwargs):
        self.max_queue_size = max_queue_size
        super().__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.max_queue_size)


class CPUPrefetcher:
    """A simple CPU-based prefetcher for data loading.

    Args:
        dataloader: The DataLoader to prefetch from.
    """

    def __init__(self, dataloader):
        self.original_loader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        """Fetches the next batch or None if exhausted."""
        try:
            return next(self.iterator)
        except StopIteration:
            return None

    def reset(self):
        """Resets the iterator to the beginning."""
        self.iterator = iter(self.original_loader)


class CUDAPrefetcher:
    """A CUDA-based prefetcher for asynchronous data loading to GPU.

    Reference: https://github.com/NVIDIA/apex/issues/304#
    Note: May increase GPU memory usage.

    Args:
        dataloader: The DataLoader to prefetch from.
        options: Configuration options with device settings.
    """

    def __init__(self, dataloader, options):
        self.original_loader = dataloader
        self.iterator = iter(dataloader)
        self.options = options
        self.device = torch.device(
            f"cuda:{options.devices[0]}" if options.devices else "cpu"
        )
        self.stream = torch.cuda.Stream(self.device)
        self._preload()

    def _preload(self):
        """Loads the next batch into GPU memory asynchronously."""
        try:
            self.current_batch = next(self.iterator)
        except StopIteration:
            self.current_batch = None
            return
        with torch.cuda.stream(self.stream):
            for key, value in self.current_batch.items():
                if torch.is_tensor(value):
                    self.current_batch[key] = value.to(
                        device=self.device, non_blocking=True
                    )

    def next(self):
        """Returns the current batch and preloads the next."""
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.current_batch
        self._preload()
        return batch

    def reset(self):
        """Resets the iterator and preloads the first batch."""
        self.iterator = iter(self.original_loader)
        self._preload()


class PrefetcherIterator:
    """An iterable wrapper for prefetchers, supporting enumeration.

    Args:
        prefetcher: Either a CPUPrefetcher or CUDAPrefetcher instance.
        length (int, optional): Total number of batches.
    """

    def __init__(self, prefetcher, length=None):
        self.prefetcher = prefetcher
        self.length = length

    def __iter__(self):
        self.prefetcher.reset()
        batch = self.prefetcher.next()
        indices = range(self.length) if self.length is not None else iter(int, 1)

        for idx in indices:
            if batch is None:
                break
            yield batch
            batch = self.prefetcher.next()

    def __len__(self):
        return self.length if self.length is not None else 0


class ImageDataset(data.Dataset):
    """A dataset class for loading paired LR and HR images with memory prefetching.

    Args:
        phase (str): Dataset phase ('train' or 'valid').
        options: Configuration options.
        load_to_memory (bool): Whether to preload main dataset to memory (default: True).
        load_extra_to_memory (bool): Whether to preload extra datasets to memory (default: False).
    """

    def __init__(self, phase: str, options):
        self.phase = phase
        self.options = options
        self.scale = options.model["upscaling_factor"]
        self.load_to_memory = options.load_to_memory
        self.load_extra_to_memory = options.load_extra_to_memory

        # Main data paths
        self.hr_path = (
            os.path.join(options.data_path, "HR")
            if phase == "train"
            else "SR_Test/Set5/HR"
        )
        self.lr_path = (
            os.path.join(options.data_path, f"x{self.scale}")
            if phase == "train"
            else f"SR_Test/Set5/x{self.scale}"
        )

        # Extra data paths from opt.extra_data
        self.extra_hr_paths: List[str] = []
        self.extra_lr_paths: List[str] = []
        if hasattr(options, "extra_data") and options.extra_data:
            for extra_path in options.extra_data:
                self.extra_hr_paths.append(
                    os.path.join(extra_path, "HR")
                    if phase == "train"
                    else "SR_Test/Set5/HR"
                )
                self.extra_lr_paths.append(
                    os.path.join(extra_path, f"x{self.scale}")
                    if phase == "train"
                    else f"SR_Test/Set5/x{self.scale}"
                )

        # Collect all image names
        self.image_names: List[Tuple[str, str, str]] = []  # (name, hr_path, lr_path)
        self._collect_image_names()

        self.repeat = options.repeat if phase == "train" else 1
        self.memory_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Load data to memory if enabled
        self._preload_to_memory()

    def _collect_image_names(self):
        """Collects image names from all data paths."""
        # Main dataset
        main_images = os.listdir(self.hr_path)
        for img_name in main_images:
            self.image_names.append((img_name, self.hr_path, self.lr_path))

        # Extra datasets
        for hr_path, lr_path in zip(self.extra_hr_paths, self.extra_lr_paths):
            extra_images = os.listdir(hr_path)
            for img_name in extra_images:
                self.image_names.append((img_name, hr_path, lr_path))

    def _load_image(self, path: str) -> np.ndarray:
        """Loads an image in RGB format using OpenCV."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]  # BGR to RGB
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img

    def _preload_image(
        self, img_name: str, hr_path: str, lr_path: str
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """Loads a single image pair for preloading."""
        hr_img = self._load_image(os.path.join(hr_path, img_name))
        lr_img = self._load_image(os.path.join(lr_path, img_name))
        return img_name, lr_img, hr_img

    def _preload_to_memory(self):
        """Preloads images to memory using parallel loading."""
        if not self.load_to_memory and not self.load_extra_to_memory:
            return

        tasks = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            for img_name, hr_path, lr_path in self.image_names:
                # Load main dataset if enabled, extra datasets if load_extra_to_memory is True
                is_main_dataset = hr_path == self.hr_path
                should_load = (is_main_dataset and self.load_to_memory) or (
                    not is_main_dataset and self.load_extra_to_memory
                )
                if should_load:
                    tasks.append(
                        executor.submit(self._preload_image, img_name, hr_path, lr_path)
                    )

            # Collect results
            for future in tasks:
                try:
                    img_name, lr_img, hr_img = future.result()
                    self.memory_cache[img_name] = (lr_img, hr_img)
                except Exception as e:
                    print(f"Warning: Failed to preload {img_name}: {e}")

    def _get_patch(
        self, lr_img: np.ndarray, hr_img: np.ndarray, patch_size: int, scale: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts matching patches from LR and HR images."""
        lr_h, lr_w = lr_img.shape[:2]
        lr_patch_size = patch_size // scale

        lr_x = random.randrange(0, lr_w - lr_patch_size + 1)
        lr_y = random.randrange(0, lr_h - lr_patch_size + 1)
        hr_x, hr_y = scale * lr_x, scale * lr_y

        lr_patch = lr_img[lr_y : lr_y + lr_patch_size, lr_x : lr_x + lr_patch_size]
        hr_patch = hr_img[hr_y : hr_y + patch_size, hr_x : hr_x + patch_size]

        return lr_patch, hr_patch

    def _augment(
        self, lr_img: np.ndarray, hr_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies random augmentation (rotation, flips, and blurs)."""
        # Geometric transforms
        if random.random() < 0.5:
            rotations = random.randint(0, 3)
            lr_img, hr_img = map(lambda x: np.rot90(x, rotations), (lr_img, hr_img))

        if random.random() < 0.5:
            lr_img, hr_img = map(np.flipud, (lr_img, hr_img))

        if random.random() < 0.5:
            lr_img, hr_img = map(np.fliplr, (lr_img, hr_img))

        return lr_img, hr_img

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Returns a single data sample as a dictionary."""
        idx = index % len(self.image_names)
        img_name, hr_path, lr_path = self.image_names[idx]

        # Load from memory if available, otherwise from disk
        if img_name in self.memory_cache:
            lr_img, hr_img = self.memory_cache[img_name]
            lr_img, hr_img = lr_img.copy(), hr_img.copy()  # Avoid modifying cached data
        else:
            hr_img = self._load_image(os.path.join(hr_path, img_name))
            lr_img = self._load_image(os.path.join(lr_path, img_name))

        if self.phase == "train":
            lr_img, hr_img = self._get_patch(
                lr_img, hr_img, self.options.img_size, self.scale
            )
            lr_img, hr_img = self._augment(lr_img, hr_img)
        else:
            lr_h, lr_w = lr_img.shape[:2]
            hr_img = hr_img[: lr_h * self.scale, : lr_w * self.scale]

        lr_tensor = (
            torch.from_numpy(np.ascontiguousarray(lr_img)).permute(2, 0, 1).float()
        )
        hr_tensor = (
            torch.from_numpy(np.ascontiguousarray(hr_img)).permute(2, 0, 1).float()
        )

        return {"lr_image": lr_tensor / 255.0, "hr_image": hr_tensor / 255.0}

    def __len__(self) -> int:
        return len(self.image_names) * self.repeat


def get_dataloader(options):
    """Creates and returns prefetching data iterators and raw dataloaders."""
    train_dataset = ImageDataset("train", options)
    valid_dataset = ImageDataset("valid", options)

    train_dataloader = PrefetchDataLoader(
        max_queue_size=4,
        dataset=train_dataset,
        batch_size=options.train_batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_dataloader = PrefetchDataLoader(
        max_queue_size=4,
        dataset=valid_dataset,
        batch_size=options.valid_batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        pin_memory=True,
    )

    prefetcher_cls = (
        CUDAPrefetcher
        if options.devices and torch.cuda.is_available()
        else CPUPrefetcher
    )
    train_prefetcher = (
        prefetcher_cls(train_dataloader, options)
        if prefetcher_cls == CUDAPrefetcher
        else prefetcher_cls(train_dataloader)
    )
    valid_prefetcher = (
        prefetcher_cls(valid_dataloader, options)
        if prefetcher_cls == CUDAPrefetcher
        else prefetcher_cls(valid_dataloader)
    )

    train_iterator = PrefetcherIterator(
        train_prefetcher, len(train_dataset) // options.train_batch_size
    )
    valid_iterator = PrefetcherIterator(
        valid_prefetcher, len(valid_dataset) // options.valid_batch_size
    )

    return train_iterator, valid_iterator, train_dataloader, valid_dataloader


if __name__ == "__main__":
    options = get_option()
    train_iter, valid_iter, train_dl, valid_dl = get_dataloader(options)

    print("Using prefetch iterator:")
    for i, batch in enumerate(train_iter):
        print(f"Batch {i}: LR {batch['lr_image'].shape}, HR {batch['hr_image'].shape}")
        print(
            f"LR range: {batch['lr_image'].max():.2f} - {batch['lr_image'].min():.2f}"
        )
        if i >= 2:
            break

    print("\nUsing raw DataLoader:")
    for i, batch in enumerate(train_dl):
        print(f"Batch {i}: LR {batch['lr_image'].shape}, HR {batch['hr_image'].shape}")
        print(
            f"LR range: {batch['lr_image'].max():.2f} - {batch['lr_image'].min():.2f}"
        )
        if i >= 2:
            break
