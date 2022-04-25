import kaggle
import zipfile
import tqdm
import os

def download_dataset():
    kaggle.api.authenticate()
    print("Pobieranie...")
    kaggle.api.dataset_download_files("mateuszbuda/lgg-mri-segmentation", path = os.getcwd(), quiet =False)

    print("Rozpakowywanie...")
    with zipfile.ZipFile("lgg-mri-segmentation.zip", 'r') as zip_ref:
        for file in tqdm.tqdm(iterable = zip_ref.namelist(), total=len(zip_ref.namelist())):
            zip_ref.extract(member = file, path = "dataset/brain_tumor_segmentation")

    #os.remove("lgg-mri-segmentation.zip")