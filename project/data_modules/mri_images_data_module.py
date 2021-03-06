from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split


class MRIImagesDataModule(pl.LightningDataModule):
    def __init__(self, dataset, dataset_downloader, batch_size=10, train_test_ratio=0.9, val_size=0.1, num_workers=4, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.dataset_downloader = dataset_downloader
        self.batch_size = batch_size
        self.train_test_ratio = train_test_ratio
        self.val_train_ratio = val_size
        self.num_workers = num_workers
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        self.dataset_downloader.prepare_dataset()

    def setup(self, stage: Optional[str] = None) -> None:
        train_size = int(len(self.dataset) * self.train_test_ratio)
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])

        val_size = int(train_size * self.val_train_ratio)
        train_size = train_size - val_size
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                          num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers)
