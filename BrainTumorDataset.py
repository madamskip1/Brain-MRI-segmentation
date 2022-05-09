import os

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from BrainTumorDatasetDownloader import BrainTumorDatasetDownloader


class BrainTumorDataset(Dataset):
    MASK_MEAN = 0.0102
    MASK_STD = 0.1005
    IMAGE_MEAN = [0.0913, 0.0828, 0.0869]
    IMAGE_STD = [0.1349, 0.1234, 0.1288]
    IMAGE_CHANNEL = 1  # -1 to wszystkie

    def __init__(self, images_path=BrainTumorDatasetDownloader.IMAGES_PATH,
                 masks_path=BrainTumorDatasetDownloader.MASKS_PATH):
        self.images_path = images_path
        self.masks_path = masks_path
        self.images_num = len(os.listdir(self.images_path))

        image_mean = BrainTumorDataset.IMAGE_MEAN
        image_std = BrainTumorDataset.IMAGE_STD
        if BrainTumorDataset.IMAGE_CHANNEL != -1:
            image_mean = BrainTumorDataset.IMAGE_MEAN[BrainTumorDataset.IMAGE_CHANNEL]
            image_std = BrainTumorDataset.IMAGE_MEAN[BrainTumorDataset.IMAGE_CHANNEL]

        self.transform_image = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std)
            ]
        )

        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=BrainTumorDataset.MASK_MEAN, std=BrainTumorDataset.MASK_STD)
            ]
        )

    def __len__(self):
        return self.images_num

    def __getitem__(self, index):
        image_name = self.images_path + "image_" + str(index) + ".tif"
        mask_name = self.masks_path + "mask_" + str(index) + ".tif"
        image = Image.open(image_name)
        mask = Image.open(mask_name)

        if BrainTumorDataset.IMAGE_CHANNEL != -1:
            image_channels = image.split()
            image = image_channels[BrainTumorDataset.IMAGE_CHANNEL]

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask
