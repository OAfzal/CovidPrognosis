"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
from argparse import ArgumentParser
from typing import Callable, List, Optional, Union

import covidprognosis as cp
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms

class TwoImageDataset(torch.utils.data.Dataset):
    """
    Wrapper for returning two augmentations of the same image.

    Args:
        dataset: Pre-initialized data set to return multiple samples from.
    """

    def __init__(self, dataset: cp.data.BaseDataset):
        assert isinstance(dataset, cp.data.BaseDataset)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # randomness handled via the transform objects
        # this requires the transforms to sample randomness from the process
        # generator
        item0 = self.dataset[idx]
        item1 = self.dataset[idx]

        sample = {
            "image0": item0["image"],
            "image1": item1["image"],
            "label": item0["labels"],
        }

        return sample


def fetch_dataset(
    dataset_name: str,
    dataset_dir: Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]],
    split: str,
    transform: Optional[Callable],
    two_image: bool = False,
    label_list="all",
):
    """Dataset fetcher for config handling."""

    assert split in ("train", "val", "test")
    dataset: Union[cp.data.BaseDataset, TwoImageDataset]

    dataset = cp.data.kaggleDataset("/content/"+split)

    return dataset


def worker_init_fn(worker_id):
    """Handle random seeding."""
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed % (2 ** 32 - 1)  # pylint: disable=no-member

    np.random.seed(seed)


class XrayDataModule(pl.LightningDataModule):
    """
    X-ray data module for training models with PyTorch Lightning.

    Args:
        dataset_name: Name of the dataset.
        dataset_dir: Location of the data.
        label_list: Labels to load for training.
        batch_size: Training batch size.
        num_workers: Number of workers for dataloaders.
        use_two_images: Whether to return two augmentations of same image from
            dataset (for MoCo pretraining).
        train_transform: Transform for training loop.
        val_transform: Transform for validation loop.
        test_transform: Transform for test loop.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_dir: Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]],
        label_list: Union[str, List[str]] = "all",
        batch_size: int = 1,
        num_workers: int = 4,
        use_two_images: bool = False,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform_train = transforms.Compose([
            transforms.CenterCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        transform_test = transforms.Compose([
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        self.train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir+"train"),transform= transform_train)
        self.train_dataset,self.val_dataset = torch.utils.data.random_split(self.train_dataset,[len(self.train_dataset*0.80),len(self.train_dataset*0.20)])
        self.test_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir+"tesst"),transform = transform_test)


    def __dataloader(self, split: str) -> torch.utils.data.DataLoader:
        assert split in ("train", "val", "test")
        shuffle = False
        if split == "train":
            dataset = self.train_dataset
            shuffle = True
        elif split == "val":
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
        )

        return loader

    def train_dataloader(self):
        return self.__dataloader(split="train")

    def val_dataloader(self):
        return self.__dataloader(split="val")

    def test_dataloader(self):
        return self.__dataloader(split="test")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--dataset_name", default="mimic", type=str)
        parser.add_argument("--dataset_dir", default=None, type=str)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--num_workers", default=4, type=int)

        return parser
