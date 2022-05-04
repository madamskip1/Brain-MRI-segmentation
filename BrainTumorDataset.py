import os

from torch.utils.data import Dataset
from PIL import Image

from BrainTumorDatasetDownloader import BrainTumorDatasetDownloader


class BrainTumorDataset(Dataset):
    def __init__(self):
        self.images_num = len(os.listdir(BrainTumorDatasetDownloader.IMAGES_PATH))
        self.transform_image = None
        self.transform_mask = None

    def __len__(self):
        return self.images_num

    def __getitem__(self, index):
        image_name = BrainTumorDatasetDownloader.IMAGES_PATH + "image_" + index + ".tif"
        mask_name = BrainTumorDatasetDownloader.MASKS_PATH + "mask_" + index + ".tif"
        image = Image.open(image_name)
        _, image, _ = image.split()
        mask = Image.open(mask_name)

        if self.transform_image is not None:
            image = self.transform_image(image)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)

        return image, mask
