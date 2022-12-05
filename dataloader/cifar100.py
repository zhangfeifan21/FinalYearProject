import os
import numpy as np
import torch
import torch.nn.functional as F

from torchvision import datasets, transforms


class DatasetSplitter(torch.utils.data.Dataset):
    """
    Makes sure always using the same training/validation split
    """
    def __init__(self, parent_dataset, split_start=-1, split_end=-1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert split_start <= len(parent_dataset) - 1 and len(
            parent_dataset) >= split_end > split_start, "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start

    def __getitem__(self, index):
        assert index < len(self), "index out of bounds in split_dataset"
        return self.parent_dataset[index + self.split_start]


def get_cifar100_dataloaders(args, validation_split=0.0, max_threads=10):
    """
    Creates augmented train, validation, and test data loaders.
    :param args:
    :param validation_split:
    :param max_threads: at least 2 threads are needed (for train and val), default 10
    :return:
    """
    # TODO: normalize matrix
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    '''
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    '''
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    full_dataset = datasets.CIFAR100('_dataset', train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100('_dataset', train=False, transform=test_transform, download=False)

    # At least 2 threads are needed (for train and val)
    max_threads = 2 if max_threads < 2 else max_threads
    if max_threads >= 6:
        val_threads = 2
        train_threads = max_threads - val_threads
    else:
        val_threads = 1
        train_threads = max_threads - 1

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0 - validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset, split_end=split)
        val_dataset = DatasetSplitter(full_dataset, split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            # num_workers=train_threads,
            num_workers=0,
            pin_memory=True,  # 拷贝到CUDA中到固定内存中
            shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.batch_size,
            # num_workers=val_threads,
            num_workers=0,
            pin_memory=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            # num_workers=max_threads,
            num_workers=0,
            pin_memory=True,
            shuffle=True
        )

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader
