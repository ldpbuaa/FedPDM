import os
import glob
import pickle
import functools
import copy

import torch
import torchvision as tv
import numpy as np
from PIL import Image
from torch.utils.data import Subset, RandomSampler
import torchvision.transforms as transforms
from .fixmatchtransform import FixMatchTransform
import random

from .info import INFO
from ..pretty import log

def bin_index(dataset, name):
    if name == 'isic':
        data = dataset.samples
    else:
        data = dataset
    bins = {}
    for i, label in enumerate(data.targets):
        bins.setdefault(int(label), []).append(i)
    flattened = []
    for k in sorted(bins):
        flattened += bins[k]
    return bins, flattened


def augment(name, train, data_transform):
    if name == "FashionMNIST":
        name = "fashionmnist"
    else:
        name = name.lower()
    info = INFO[name]
    mean, std = info['moments']
    if not train:
        if name in ['isic', 'office']:
            transform_test = transforms.Compose([
            transforms.Resize(info['shape'][1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        else:
            transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        return transform_test
    else:
        augments = []
        crop_size = info['shape'][-1]
        if name in ['cifar10', 'cifar100']:
            augments.append(tv.transforms.RandomCrop(crop_size, padding=4))
            augments.append(tv.transforms.RandomHorizontalFlip())
        if name == 'svhn':
            augments.append(tv.transforms.RandomCrop(crop_size, padding=4))
        if name == 'fashionmnist':
            augments.append(tv.transforms.RandomHorizontalFlip())
        if name in ['isic', 'office']:
            augments.append(tv.transforms.Resize(160))
            augments.append(tv.transforms.RandomCrop(crop_size))
            augments.append(tv.transforms.RandomHorizontalFlip())

        augments.append(tv.transforms.ToTensor())
        augments.append(tv.transforms.Normalize(*info['moments']))
        if data_transform == 'default':
            return tv.transforms.Compose(augments)
        if data_transform == 'twice':
            return TwiceTransform(tv.transforms.Compose(augments))
        if data_transform == 'dual':
            resize = 160 if name in ['isic', 'office'] else None
            return FixMatchTransform(mean, std, crop_size, resize)


def get_stats(datasets, num_clients, num_classes):
    stats = {c:[] for c in range(num_clients)}
    for c, ds in enumerate(datasets):
        stats[c] = np.bincount(ds.targets, minlength=num_classes).tolist()
    return stats


def split(name, dataset, policy, num_clients, num_shards, alpha, beta, batch_size, drop_last, seed):
    # guarantee determinism
    np.random.seed(seed)
    torch.manual_seed(seed)
    bins, flattened = bin_index(dataset, name)
    num_classes = INFO[name.lower()]['model_params']['num_classes']

    if policy == 'iid':
        client_indices = [[] for _ in range(num_clients)]
        statistics = {c:[] for c in range(num_clients)}
        for k, idx_k in bins.items():
            np.random.shuffle(idx_k)
            for c, (idx_j, idx) in enumerate(
                        zip(client_indices, np.split(np.array(idx_k), num_clients))):
                idx_j += idx.tolist()
                statistics[c].append(len(idx))
        datasets = [IdxSubset(dataset, client_indices[c]) for c in range(num_clients)]
        for c, v in statistics.items():
            log.debug(f'client: {c}, total: {int(np.sum(v))}, data dist: {v}')
        return datasets, statistics

    if policy == 'size':
        splits = np.random.random(num_clients)
        splits *= len(dataset) / np.sum(splits)
        splits = splits.astype(np.int)
        remains = sum(splits)
        remains = np.random.randint(0, num_clients, len(dataset) - remains)
        for n in range(num_clients):
            splits[n] += sum(remains == n)
        return torch.utils.data.dataset.random_split(dataset, splits.tolist())

    if policy == 'dirichlet':
        data_num, _counter = 0, 0
        num_data = len(flattened)
        statistics = {c:[] for c in range(num_clients)}
        min_size = batch_size if drop_last else 5
        while data_num < min_size:
            client_indices = [[] for _ in range(num_clients)]
            for k, idx_k in bins.items():
                np.random.shuffle(idx_k)
                prop = np.random.dirichlet(np.repeat(alpha, num_clients))
                prop = np.array([p * (len(idx_c) < num_data / num_clients)
                                    for p, idx_c in zip(prop, client_indices)])
                prop = prop / prop.sum()
                prop = (np.cumsum(prop)*len(idx_k)).astype(int)[:-1]
                for c, (idx_j, idx) in enumerate(zip(client_indices, np.split(idx_k, prop))):
                    idx_j += idx.tolist()
                data_num = min([len(idx_c) for idx_c in client_indices])
            _counter += 1
            if _counter == 1000:
                raise "data partition is not feasible..."
        idx_subset = IdxSubset if name not in ['isic', 'office'] else ISICIdxSubset
        datasets = [idx_subset(dataset, client_indices[c]) for c in range(num_clients)]
        statistics = get_stats(datasets, num_clients, num_classes)
        for c, v in statistics.items():
            log.debug(f'client: {c}, total: {int(np.sum(v))}, data dist: {v}')
        return datasets, statistics

    if policy == 'task':
        bins, flattened = bin_index(dataset, name)
        datasets = []
        client_indices = [[] for _ in range(num_clients)]
        statistics = {c:[] for c in range(num_clients)}
        if num_shards % num_clients:
            raise ValueError(
                'Expect the number of shards to be '
                'evenly distributed to clients.')
        num_client_shards = num_shards // num_clients
        shard_size = len(dataset) // num_shards
        shards = list(range(num_shards))
        np.random.shuffle(shards)  # fix np.ramdom error
        for i in range(num_clients):
            shard_offset = i * num_client_shards
            indices = []
            for s in shards[shard_offset:shard_offset + num_client_shards]:
                if s == len(shards) - 1:
                    indices += flattened[s * shard_size:]
                else:
                    indices += flattened[s * shard_size:(s + 1) * shard_size]
            subset = Subset(dataset, indices)
            datasets.append(subset)
        return datasets, statistics

    raise TypeError(f'Unrecognized split policy {policy!r}.')


def search_classes(bins, index):
    for k, v in bins.items():
        if index in v:
            return k
    print('unknown data index!')


def init_isic_sub(name, data_dir, train):
    split = 'train' if train else 'test'
    data_path = os.path.join(data_dir, name.upper(), split)
    # transforms = ISICTransform['train' if train else 'valid']
    transforms = ISICTransform['train' if train else 'valid']
    dataset = tv.datasets.ImageFolder(data_path, transform = transforms)
    # dataset = IdxISIC(data_path)
    num_classes = INFO[name]['model_params']['num_classes']
    dataset.classes = list(range(num_classes))
    return dataset

class ISICIdxSubset(tv.datasets.ImageFolder):
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = indices
        # self.samples = self.dataset.samples[indices]
        if self.indices is not None:
            self.samples = [self.dataset.samples[i] for i in indices]
        else:
            self.samples = self.dataset.samples
        self.targets = [t for _, t in self.samples]
        self.loader = dataset.loader
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        target = torch.tensor(int(target))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def __len__(self):
        return len(self.samples)

    def __reinit_samples__(self,):
        pass


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
        transforms.Resize(240),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
valid_transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
ISICTransform = {'train':train_transform,
                        'valid':valid_transform}

def ISICDataset(
        name, train, batch_size, num_clients, num_shards,
        split_mode, parallel=False, alpha=0.5, beta=2, data_dir=None, subset=None,
        num_labeled=0, unlabeled_mu=1,
        drop_last=False, data_transform='default', seed=0):
    dataset = init_isic_sub(name, data_dir, train)
    kwargs = {'transform': augment(name, train, data_transform),
              'drop_last': drop_last}
    if parallel:
        kwargs = {'pin_memory': False, 'num_workers': 0, 'drop_last':drop_last, }
    Loader = functools.partial(torch.utils.data.DataLoader, **kwargs)
    if not train:
        return Loader(ISICIdxSubset(dataset), batch_size, train)
    dataloaders = [[], []] # unlabeled and labeled dataloaders
    bzs = [unlabeled_mu*batch_size, batch_size]
    unlabeled_dataset, labeled_dataset = split_labeled_set(name, dataset, num_labeled)
    for i, dataset in enumerate([unlabeled_dataset, labeled_dataset]):
        if dataset:
            splits, stats = split(name, dataset, split_mode, num_clients,
                            num_shards, alpha, beta, batch_size, drop_last, seed)
            for c in range(num_clients):
                loader = Loader(splits[c], bzs[i], train)
                loader.stats = stats[c]
                dataloaders[i].append(loader)
    return dataloaders[0], dataloaders[1]

def init_office_sub(name, data_dir, train, data_transform):
    split = 'train' if train else 'test'
    data_path = os.path.join(data_dir, 'Office-Caltech10', split)
    # transforms = ISICTransform['train' if train else 'valid']
    transforms = augment(name, train, data_transform)
    dataset = tv.datasets.ImageFolder(data_path, transform = transforms)
    # dataset = IdxISIC(data_path)
    num_classes = INFO[name]['model_params']['num_classes']
    dataset.classes = list(range(num_classes))
    return dataset

def OfficeDataset(
        name, train, batch_size, num_clients, num_shards,
        split_mode, parallel=False, alpha=0.5, beta=2, data_dir=None, subset=None,
        num_labeled=0, unlabeled_mu=1,
        drop_last=False, data_transform='default', seed=0):
    dataset = init_office_sub(name, data_dir, train, data_transform)
    # transform = augment(name, train, data_transform)
    kwargs = {'drop_last': drop_last}
    if parallel:
        kwargs = {'pin_memory': False, 'num_workers': 0, 'drop_last':drop_last, }
    Loader = functools.partial(torch.utils.data.DataLoader, **kwargs)
    if not train:
        return Loader(ISICIdxSubset(dataset), batch_size, train)
    dataloaders = [[], []] # unlabeled and labeled dataloaders
    bzs = [unlabeled_mu*batch_size, batch_size]
    unlabeled_dataset, labeled_dataset = split_labeled_set(name, dataset, num_labeled)
    for i, dataset in enumerate([unlabeled_dataset, labeled_dataset]):
        if dataset:
            splits, stats = split(name, dataset, split_mode, num_clients,
                            num_shards, alpha, beta, batch_size, drop_last, seed)
            for c in range(num_clients):
                loader = Loader(splits[c], bzs[i], train)
                loader.stats = stats[c]
                dataloaders[i].append(loader)
    return dataloaders[0], dataloaders[1]


def Dataset(
        name, train, batch_size, num_clients, num_shards,
        split_mode, parallel=False, alpha=0.5, beta=2, data_dir=None, subset=None,
        num_labeled=0, unlabeled_mu=1,
        drop_last=False, data_transform='default', seed=0):
    assert data_transform in ['default', 'twice', 'dual'], \
                                f'Undefined data transform: {data_transform}!'
    # dataset creation
    # FIXME messy name config
    if name == 'fashionmnist':
        name = 'FashionMNIST'
    elif name in ['cifar10', 'cifar100', 'mnist', 'svhn', 'emnist']:
        name = name.upper()
    cls = getattr(tv.datasets, name)
    path = os.path.join(data_dir, name.lower())
    kw = { 'root': path,
        'train': train,
        'transform': augment(name, train, data_transform)}
    if name == 'SVHN':
        kw.pop('train')
        kw['split'] = 'train' if train else 'test'
    if name == 'EMNIST':
        kw['split'] = 'balanced'
    try:
        dataset = cls(**kw, download=False)
    except RuntimeError:
        dataset = cls(**kw, download=True)
    try:
        targets = copy.deepcopy(dataset.targets)
    except:
        targets = copy.deepcopy(dataset.labels) # svhn
    dataset.targets = targets
    kwargs = {'drop_last': drop_last}
    # if True:
    #     kwargs = {'pin_memory': False, 'num_workers': 1, 'drop_last':drop_last, }
    if parallel:
       kwargs = {'pin_memory': False, 'num_workers': 0, 'drop_last':drop_last, }
    Loader = functools.partial(torch.utils.data.DataLoader, **kwargs)
    if not train:
        # only return a subset of test images for visulization purpose
        indices = list(range(len(dataset.targets)))
        dataset.data_indices = indices
        if subset:
            indices = subset
        # otherwise return entire testset
        return Loader(IdxSubset(dataset, indices), batch_size, train)
    dataloaders = [[], []] # unlabeled and labeled dataloaders
    bzs = [unlabeled_mu*batch_size, batch_size]
    unlabeled_dataset, labeled_dataset = split_labeled_set(name, dataset, num_labeled)
    for i, dataset in enumerate([unlabeled_dataset, labeled_dataset]):
        if dataset:
            splits, stats = split(name, dataset, split_mode, num_clients,
                            num_shards, alpha, beta, batch_size, drop_last, seed)
            for c in range(num_clients):
                loader = Loader(splits[c], bzs[i], train)
                loader.stats = stats[c]
                dataloaders[i].append(loader)
    return dataloaders[0], dataloaders[1]

def split_labeled_set(name, train_dataset, num_labeled):
    """ split a labeled set if split_labled is True
    """
    num_classes = len(np.unique(train_dataset.targets))
    num_labeled_cls = num_labeled // num_classes
    data_indices = {} # the data indices for entire trainset
    classes = list(range(num_classes))
    for j in classes:
        data_indices[j] = [i for i, label in
                        enumerate(train_dataset.targets) if label == j]
    flatten_indices = list(range(len(train_dataset.targets)))
    idx_val, idx_train = [], []
    for cls_idx, img_ids in data_indices.items():
        np.random.shuffle(img_ids)
        idx_val.extend(img_ids[:num_labeled_cls])
        idx_train.extend(img_ids[num_labeled_cls:])
    train_data, val_data = [], []
    if num_labeled > 0:
        if name not in ['isic', 'office']:
            train_data = copy.deepcopy(train_dataset)
            train_data.data = np.delete(train_dataset.data, idx_val, axis=0)
            train_data.targets = np.delete(train_dataset.targets, idx_val, axis=0)
            train_data.data_indices = np.array(idx_train)
            val_data = copy.deepcopy(train_dataset)
            val_data.data = np.delete(train_dataset.data, idx_train, axis=0)
            val_data.targets = np.delete(train_dataset.targets, idx_train, axis=0)
            val_data.data_indices = np.array(idx_val)
        else:
            train_data = copy.deepcopy(train_dataset)
            train_data.imgs = np.delete(train_dataset.imgs, idx_val, axis=0)
            train_data.samples = np.delete(train_dataset.samples, idx_val, axis=0)
            train_data.targets = np.delete(train_dataset.targets, idx_val, axis=0)
            train_data.data_indices = np.array(idx_train)
            val_data = copy.deepcopy(train_dataset)
            train_data.imgs = np.delete(train_dataset.imgs, idx_val, axis=0)
            val_data.samples = np.delete(train_dataset.samples, idx_train, axis=0)
            val_data.targets = np.delete(train_dataset.targets, idx_train, axis=0)
            val_data.data_indices = np.array(idx_val)
    else:
        train_data = copy.deepcopy(train_dataset)
        train_data.data_indices = np.array(flatten_indices)
    return train_data, val_data

def split_toy(train_dataset, num_classes, num_sub_classes):
    """ split a toy dataset from trainset
        that only contains several sub classes
    """
    sub_classes = list(range(num_sub_classes))
    print(f'sampled sub classes: {sub_classes}')
    data_indices = {}
    for j in sub_classes:
        data_indices[j] = [i for i, label in
                        enumerate(train_dataset.targets) if label == j]
    idx_train = []
    for cls_idx, img_ids in data_indices.items():
        idx_train.extend(img_ids)
    train_data = copy.deepcopy(train_dataset)
    train_data.data = train_dataset.data[idx_train]
    train_data.targets = np.array(train_dataset.targets)[idx_train]
    train_data.data_indices = np.array(idx_train)
    train_data.classes = [train_dataset.classes[i]
                                for i in sub_classes]
    return train_data


class IdxSubset(torch.utils.data.Dataset):
    """sub dataset that additionally returns image indices"""
    def __init__(self, dataset, indices=None):
        if indices is not None:
            self.data = self._split_sub(dataset.data, indices)
            self.targets = self._split_sub(dataset.targets, indices)
            self.data_indices = self._split_sub(dataset.data_indices, indices)
        else:
            self.data = dataset.data
            self.targets = dataset.targets
            self.data_indices = dataset.data_indices
        self.transform = dataset.transform
        # self.classes = dataset.classes
        # print('dataset len:', len(self.data), len(self.targets), len(self.data_indices))

    def _split_sub(self, input, indices):
        data = np.array(copy.deepcopy(input))
        # print(f'debug inputs: {data[indices]}')
        return data[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target, indices = \
            self.data[idx], self.targets[idx], self.data_indices[idx]
        img = np.array(img)
        if img.shape[0] <= 3:  # for svhn: [3, 32, 32]
            img = img.transpose(1,2,0)
        img = Image.fromarray(np.array(img))


        if self.transform is not None:
            img = self.transform(img)
        return img, target, indices



class CustomDataset(torch.utils.data.Dataset):
    """create a dataset with input data."""
    def __init__(self, images, labels, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform:
            sample = self.transform(sample)

        return sample

class TwiceTransform:
    """
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return [out1, out2]