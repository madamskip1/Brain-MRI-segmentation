from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split

from BrainTumorDataset import BrainTumorDataset
from BrainTumorDatasetDownloader import BrainTumorDatasetDownloader


class MRIImagesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, train_test_ratio=0.9, val_size=0.1):
        super().__init__()
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.image_folder = None
        self.batch_size = batch_size
        self.train_test_ratio = train_test_ratio
        self.val_train_ratio = val_size

    def prepare_data(self) -> None:
        dataset_downloader = BrainTumorDatasetDownloader()
        dataset_downloader.prepare_dataset()

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = BrainTumorDataset()

        train_size = int(len(dataset) * self.train_test_ratio)
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        val_size = int(train_size * self.val_train_ratio)
        train_size = train_size - val_size
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)
