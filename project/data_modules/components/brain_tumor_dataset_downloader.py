import os
import shutil
import zipfile

import kaggle
from tqdm import tqdm


class BrainTumorDatasetDownloader:
    def __init__(self, dataset_path, images_path, masks_path, **kwargs):
        self.dataset_path = dataset_path
        self.images_path = images_path
        self.masks_path = masks_path

    def prepare_dataset(self):
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        if not os.path.exists(self.images_path) \
                and not os.path.exists(self.masks_path):
            os.mkdir(self.images_path)
            os.mkdir(self.masks_path)

            print("Rozpoczęto przygotowywanie datasetu...")
            self.__download()
            self.__unzip()
            self.__sort_dataset()
            self.__remove_temps()
            self.__standardize_names()
        else:
            print("Dataset już istnieje...")

        print("Zakończono przygotowywanie datasetu...")

    def __download(self):
        kaggle.api.authenticate()
        print("Pobieranie...")
        kaggle.api.dataset_download_files("mateuszbuda/lgg-mri-segmentation", path=os.getcwd(), quiet=False)
        print("Pobrano...")

    def __unzip(self):
        print("Rozpakowywanie...")
        with zipfile.ZipFile("lgg-mri-segmentation.zip", 'r') as zip_ref:
            for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                zip_ref.extract(member=file, path="dataset")
        print("Rozpakowano...")

    def __sort_dataset(self):
        print("Sortowanie...")
        dataset_dirs = os.listdir("dataset/kaggle_3m")
        dataset_dirs.remove("data.csv")
        dataset_dirs.remove("README.md")

        # TODO dodać tqdm
        for subdir in dataset_dirs:
            files_in_subdir = os.listdir("dataset/kaggle_3m/" + subdir)
            images = list(filter(lambda file_name: "mask" not in file_name, files_in_subdir))
            masks = list(filter(lambda file_name: "mask" in file_name, files_in_subdir))

            for image in images:
                shutil.move("dataset/kaggle_3m/" + subdir + "/" + image,
                            self.images_path + image)

            for mask in masks:
                shutil.move("dataset/kaggle_3m/" + subdir + "/" + mask,
                            self.masks_path + mask)

        print("Posortowano...")

    def __remove_temps(self):
        print("Usuwanie tymczasowych folderów...")
        shutil.rmtree("dataset/kaggle_3m")
        shutil.rmtree("dataset/lgg-mri-segmentation")
        print("Usunięto...")

    def __standardize_names(self):
        print("Standaryzowanie nazewnictwa zdjęć...")
        images = os.listdir(self.images_path)

        # TODO dodać tqdm
        for i, image_name in enumerate(images):
            dot_index = image_name.rfind('.')
            mask_name = image_name[:dot_index] + "_mask" + image_name[dot_index:]
            mask_name = self.masks_path + mask_name
            image_name = self.images_path + image_name
            new_image_name = self.images_path + "image_" + str(i) + ".tif"
            new_mask_name = self.masks_path + "mask_" + str(i) + ".tif"
            shutil.move(image_name, new_image_name)
            shutil.move(mask_name, new_mask_name)
        print("Ustandaryzowano...")
