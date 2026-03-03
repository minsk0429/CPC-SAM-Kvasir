import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from einops import repeat
from PIL import Image
from PIL import ImageFilter
import cv2
from copy import deepcopy
import itertools
from torchvision import transforms


def blur(img, p=0.5):
    """Apply Gaussian blur to PIL image"""
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        h, w = image.shape[:2]
        if h != self.output_size[0] or w != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w, 1), order=3)
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
        
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        
        # Convert to tensor
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


class Kvasir_dataset(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        
        if self.split == "train":
            with open(self._base_dir + "/train.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "").strip() for item in self.sample_list]
        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "").strip() for item in self.sample_list]
        
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        
        if self.split == "train":
            image_path = os.path.join(self._base_dir, "train/images", f"{case}.png")
            label_path = os.path.join(self._base_dir, "train/masks", f"{case}.png")
        else:
            image_path = os.path.join(self._base_dir, "test/images", f"{case}.png")
            label_path = os.path.join(self._base_dir, "test/masks", f"{case}.png")
        
        # Load image (RGB)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (grayscale)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Binarize label (0: background, 1: polyp)
        label = (label > 127).astype(np.float32)
        
        sample = {"image": image, "label": label}
        
        if self.split == "train":
            sample = self.transform(sample)
        else:
            # For validation/test, just convert to tensor without augmentation
            h, w = image.shape[:2]
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label.long()}
        
        sample['case_name'] = case
        
        return sample


class Kvasir_dataset_aug(Dataset):
    """Dataset with stronger augmentation for semi-supervised learning"""
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        
        if self.split == "train":
            with open(self._base_dir + "/train.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "").strip() for item in self.sample_list]
        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "").strip() for item in self.sample_list]
        
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        
        # Load image and mask
        if self.split == "train":
            image_path = os.path.join(self._base_dir, "train", "images", case + ".png")
            label_path = os.path.join(self._base_dir, "train", "masks", case + ".png")
        else:
            image_path = os.path.join(self._base_dir, "test", "images", case + ".png")
            label_path = os.path.join(self._base_dir, "test", "masks", case + ".png")
        
        # Load image (RGB)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (grayscale)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Binarize label
        label = (label > 127).astype(np.float32)
        
        # Apply color jitter and blur augmentation (like ACDC_dataset_aug)
        img = Image.fromarray((image * 255).astype(np.uint8))
        
        if random.random() < 0.8:
            img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
        img = blur(img, p=0.5)
        
        image = np.array(img).astype(np.float32) / 255.0
        
        sample = {"image": image, "label": label}
        
        if self.split == "train":
            sample = self.transform(sample)
        
        sample['case_name'] = case
        
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
