import os
import shutil
import zipfile
from typing import Optional

import kaggle
import pytorch_lightning as pl
import tqdm
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split

from BrainTumorDataset import BrainTumorDataset


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

        self.IMAGES_PATH = "dataset/images/"
        self.MASKS_PATH = "dataset/masks/"

    def prepare_data(self) -> None:
        MRIImagesDataModule.__download_and_unzip_dataset()

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = BrainTumorDataset(self.IMAGES_PATH, self.MASKS_PATH)

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

    @staticmethod
    def __download_and_unzip_dataset():
        print("Rozpoczęto przygotowywanie datasetu...")
        kaggle.api.authenticate()
        print("Pobieranie...")
        kaggle.api.dataset_download_files("mateuszbuda/lgg-mri-segmentation", path=os.getcwd(), quiet=False)
        print("Pobrano...")

        if not os.path.exists("dataset"):
            os.mkdir("dataset")

        images_path = "dataset/images/"
        masks_path = "dataset/masks/"

        if not os.path.exists(images_path) and not os.path.exists(masks_path):
            os.mkdir(images_path)
            os.mkdir(masks_path)

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
                                images_path + image)

                for mask in masks:
                    shutil.move("dataset/kaggle_3m/" + subdir + "/" + mask,
                                masks_path + mask)
            print("Posortowano...")

            print("Usuwanie tymczasowych folderów...")
            shutil.rmtree("dataset/kaggle_3m")
            shutil.rmtree("dataset/lgg-mri-segmentation")
            print("Usunięto...")

            print("Standaryzowanie nazewnictwa zdjęć...")

            images = os.listdir(images_path)

            for i, image_name in enumerate(images):
                dot_index = image_name.rfind('.')
                mask_name = image_name[:dot_index] + "_mask" + image_name[dot_index:]
                mask_name = masks_path + mask_name
                image_name = images_path + image_name
                new_image_name = images_path + "image_" + str(i) + ".tif"
                new_mask_name = masks_path + "mask_" + str(i) + ".tif"
                shutil.move(image_name, new_image_name)
                shutil.move(mask_name, new_mask_name)

            print("Ustandaryzowano...")
        else:
            print("Dataset już istnieje...")

        print("Zakończono przygotowywanie datasetu...")
