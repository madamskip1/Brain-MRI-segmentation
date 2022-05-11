import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    def __init__(self, images_path, masks_path, mask_mean, mask_std, image_mean, image_std, image_channel, **kwargs):
        self.images_path = images_path
        self.masks_path = masks_path
        self.images_num = len(os.listdir(self.images_path))

        self.mask_mean = mask_mean
        self.mask_std = mask_std
        self.image_mean = image_mean
        self.image_std = image_std
        self.image_channel = image_channel

        if self.image_channel != -1:
            self.image_mean = image_mean[self.image_channel]
            self.image_std = image_std[self.image_channel]

        self.transform_image = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std)
            ]
        )

        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mask_mean, std=self.mask_std)
            ]
        )

    def __len__(self):
        return self.images_num

    def __getitem__(self, index):
        image_name = self.images_path + "image_" + str(index) + ".tif"
        mask_name = self.masks_path + "mask_" + str(index) + ".tif"
        image = Image.open(image_name)
        mask = Image.open(mask_name)

        if self.image_channel != -1:
            image_channels = image.split()
            image = image_channels[self.image_channel]

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask
