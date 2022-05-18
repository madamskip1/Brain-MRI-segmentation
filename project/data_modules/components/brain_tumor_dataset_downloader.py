import os
import shutil
import zipfile
import json


import kaggle
from tqdm import tqdm


class BrainTumorDatasetDownloader:
    def __init__(self, dataset_path, images_path, masks_path, **kwargs):
        self.dataset_path = dataset_path
        self.images_path = images_path
        self.masks_path = masks_path

    def prepare_dataset(self):
        print("Rozpoczęto przygotowywanie datasetu...")

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        if not os.path.exists(self.images_path) \
                and not os.path.exists(self.masks_path):
            os.mkdir(self.images_path)
            os.mkdir(self.masks_path)

            print("Rozpoczęto przygotowywanie datasetu...")
            self.__download()
            self.__unzip()
            self.__remove_damaged_images
            self.__sort_dataset()
            self.__remove_temps()
            self.__standardize_names()
        else:
            print("Dataset już istnieje...")

        print("Zakończono przygotowywanie datasetu...")

    def __download(self):
        kaggle.api.authenticate()
        print("Pobieranie...")
        kaggle.api.dataset_download_files("mateuszbuda/lgg-mri-segmentation", path=self.dataset_path, quiet=False)
        print("Pobrano...")

    def __unzip(self):
        print("Rozpakowywanie...")
        with zipfile.ZipFile(f"{self.dataset_path}/lgg-mri-segmentation.zip", 'r') as zip_ref:
            for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                zip_ref.extract(member=file, path=self.dataset_path)
        print("Rozpakowano...")

    def __remove_damaged_images(self):
        print("Usuwanie uszkodzonych zdjęć")

        images_to_delete_file = open("images_to_delete.json")
        images_to_delete_json = json.load(images_to_delete_file)

        for images_dir in images_to_delete_json:
            dir = images_dir["dir"]
            for image_index in images_dir["files_index"]:
                image_path = dir + "/" + dir + "_" + str(image_index) + ".tif"
                mask_path = dir + "/" + dir + "_" + str(image_index) + "_mask.tif"
                os.remove(f"{image_path}")
                os.remove(f"{mask_path}")

        print("Usunięto...")

    def __sort_dataset(self):
        print("Sortowanie...")
        dataset_dirs = os.listdir(f"{self.dataset_path}/kaggle_3m")
        dataset_dirs.remove(f"data.csv")
        dataset_dirs.remove(f"README.md")

        for subdir in tqdm(iterable=dataset_dirs, total=len(dataset_dirs)):
            files_in_subdir = os.listdir(f"{self.dataset_path}/kaggle_3m/{subdir}")
            images = list(filter(lambda file_name: "mask" not in file_name, files_in_subdir))
            masks = list(filter(lambda file_name: "mask" in file_name, files_in_subdir))

            for image in images:
                shutil.move(f"{self.dataset_path}/kaggle_3m/{subdir}/{image}",
                            f"{self.images_path}/{image}")

            for mask in masks:
                shutil.move(f"{self.dataset_path}/kaggle_3m/{subdir}/{mask}",
                            f"{self.masks_path}/{mask}")

        print("Posortowano...")

    def __remove_temps(self):
        print("Usuwanie tymczasowych folderów...")
        shutil.rmtree(f"{self.dataset_path}/kaggle_3m")
        shutil.rmtree(f"{self.dataset_path}/lgg-mri-segmentation")
        print("Usunięto...")

    def __standardize_names(self):
        print("Standaryzowanie nazewnictwa zdjęć...")
        images = os.listdir(self.images_path)

        for i, image_name in enumerate(tqdm(iterable=images, total=len(images))):
            dot_index = image_name.rfind('.')
            mask_name = f"{self.masks_path}/{image_name[:dot_index]}_mask{image_name[dot_index:]}"
            image_name = f"{self.images_path}/{image_name}"
            new_image_name = f"{self.images_path}/image_{i}.tif"
            new_mask_name = f"{self.masks_path}/mask_{i}.tif"
            shutil.move(image_name, new_image_name)
            shutil.move(mask_name, new_mask_name)
        print("Ustandaryzowano...")
