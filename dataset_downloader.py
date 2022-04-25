import kaggle
import zipfile
import tqdm
import os


def download_dataset():
    kaggle.api.authenticate()
    print("Pobieranie...")
    kaggle.api.dataset_download_files("mateuszbuda/lgg-mri-segmentation", path=os.getcwd(), quiet=False)
    print("Pobrano...")

    print("Rozpakowywanie...")
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    if not os.path.exists("dataset/brain_tumor_segmentation"):
        with zipfile.ZipFile("lgg-mri-segmentation.zip", 'r') as zip_ref:
            for file in tqdm.tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                zip_ref.extract(member=file, path="dataset")

        os.rename("dataset/kaggle_3m", "dataset/brain_tumor_segmentation")
        os.remove("dataset/lgg-mri-segmentation")
        print("Rozpakowano...")

        # os.remove("lgg-mri-segmentation.zip")
    else:
        print("Dataset już istnieje...")