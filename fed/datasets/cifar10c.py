import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image



corruptions=[
"gaussian_noise",
"shot_noise",
"fog",
"frost",
"glass_blur",
"brightness",
"contrast",
]


class CorruptDataset(Dataset):
    def __init__(self, data_dir, images_file, labels_file, transform=None):
        data_dir = os.path.expanduser(data_dir)
        images_path = os.path.join(data_dir, 'cifar10c', images_file)
        labels_path = os.path.join(data_dir, 'cifar10c', labels_file)
        images = np.load(images_path)
        self.images = [Image.fromarray(img) for img in images]
        self.labels = torch.from_numpy(np.load(labels_path))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, idx
