import os
import shutil
import zipfile
from typing import Optional

import kaggle
import pytorch_lightning as pl
import tqdm
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


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
        MRIImagesDataModule.__download_and_unzip_dataset()

    def setup(self, stage: Optional[str] = None) -> None:
        self.image_folder = ImageFolder("dataset/brain_tumor_segmentation")
        print(self.image_folder.classes)
        print(len(self.image_folder.classes))
        train_size = int(len(self.image_folder) * self.train_test_ratio)
        test_size = len(self.image_folder) - train_size

        self.train_dataset, self.test_dataset = random_split(self.image_folder, [train_size, test_size])

        val_size = int(train_size * self.val_train_ratio)
        train_size = train_size - val_size

        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=4)

    @staticmethod
    def __download_and_unzip_dataset():
        kaggle.api.authenticate()
        print("Pobieranie...")
        kaggle.api.dataset_download_files("mateuszbuda/lgg-mri-segmentation", path=os.getcwd(), quiet=False)
        print("Pobrano...")


        if not os.path.exists("dataset"):
            os.mkdir("dataset")

        if not os.path.exists("dataset/brain_tumor_segmentation"):
            os.mkdir("dataset/brain_tumor_segmentation")
            os.mkdir("dataset/brain_tumor_segmentation/images")
            os.mkdir("dataset/brain_tumor_segmentation/masks")

            print("Rozpakowywanie...")
            with zipfile.ZipFile("lgg-mri-segmentation.zip", 'r') as zip_ref:
                for file in tqdm.tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                    zip_ref.extract(member=file, path="dataset")
            print("Rozpakowano...")

            print("Sortowanie...")
            dataset_dirs = os.listdir("dataset/kaggle_3m")
            dataset_dirs.remove("data.csv")
            dataset_dirs.remove("README.md")

            for subdir in dataset_dirs:
                files_in_subdir = os.listdir("dataset/kaggle_3m/" + subdir)
                images = list(filter(lambda file_name: "mask" not in file_name, files_in_subdir))
                masks = list(filter(lambda file_name: "mask" in file_name, files_in_subdir))

                for image in images:
                    shutil.move("dataset/kaggle_3m/" + subdir + "/" + image,
                                "dataset/brain_tumor_segmentation/images/" + image)

                for mask in masks:
                    shutil.move("dataset/kaggle_3m/" + subdir + "/" + mask,
                                "dataset/brain_tumor_segmentation/masks/" + mask)
            print("Posortowano...")

            print("Usuwanie tymczasowych folderów...")
            shutil.rmtree("dataset/kaggle_3m")
            shutil.rmtree("dataset/lgg-mri-segmentation")
            print("Usunięto...")
        else:
            print("Dataset już istnieje...")
