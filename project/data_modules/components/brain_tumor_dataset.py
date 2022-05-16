import os

import albumentations as A
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):
    AUGUMENTATION = True

    def __init__(self, images_path, masks_path, mask_mean, mask_std, image_mean, image_std, image_channel, **kwargs):
        self.images_path = images_path
        self.masks_path = masks_path

        self.mask_mean = mask_mean
        self.mask_std = mask_std
        self.image_mean = image_mean
        self.image_std = image_std
        self.image_channel = image_channel

        if self.image_channel != -1:
            self.image_mean = image_mean[self.image_channel]
            self.image_std = image_std[self.image_channel]

        self.transform_augumentations = self.get_augumentations_transform()
        self.transform_image = self.get_image_transforms()
        self.transform_mask = self.get_mask_transforms()

    def __len__(self):
        return len(os.listdir(self.images_path))

    def __getitem__(self, index):
        image_name = f"{self.images_path}/image_{index}.tif"
        mask_name = f"{self.masks_path}/mask_{index}.tif"
        image = Image.open(image_name)
        mask = Image.open(mask_name)

        if self.image_channel != -1:
            image_channels = image.split()
            image = image_channels[self.image_channel]

        image, mask = self.transform(image, mask)
        return image, mask

    def transform(self, image, mask):
        if BrainTumorDataset.AUGUMENTATION:
            image_np = np.array(image)
            mask_np = np.array(mask)
            transformed = self.transform_augumentations(image=image_np, mask=mask_np)
            image = Image.fromarray(transformed["image"])
            mask = Image.fromarray(transformed["mask"])

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask

    def get_image_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std)
            ]
        )

    def get_mask_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

    def get_augumentations_transform(self):
        return A.Compose(
            [
                A.Flip(p=0.2),
                A.NoOp(),
                A.Perspective(keep_size=True, p=0.2),
                A.RandomRotate90(p=0.2),
                A.Transpose(p=0.2)
            ]
        )
