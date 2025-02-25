from .image import Dataset as ImageDataset, ISICDataset, OfficeDataset
from .info import INFO
from .cifar10c import corruptions, CorruptDataset

datasets_map = {
    'mnist': ImageDataset,
    'fashionmnist': ImageDataset,
    'emnist': ImageDataset,
    'cifar10': ImageDataset,
    'cifar100': ImageDataset,
    'svhn': ImageDataset,
    'isic': ISICDataset,
    'office': OfficeDataset,
}
