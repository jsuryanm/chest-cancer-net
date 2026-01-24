import os
import zipfile 
import gdown 
import shutil 
from pathlib import Path

from src.cancer_clf.logger.logger import logger
from src.cancer_clf.utils.common import get_size
from src.cancer_clf.entity.config_entity import DataIngestionConfig
from src.cancer_clf.utils.seeds import set_seed

import random

class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config
        set_seed()

    def download_file(self) -> str: 
        '''
        Fetches the data from url
        '''
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = str(self.config.local_data_file)
            os.makedirs("artifacts/data_ingestion",exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into {zip_download_dir}")

        except Exception as e:
            raise e
    
    def extract_zip_file(self):
        '''
        zip_file_path: str
        Extract the zip file into the data directory
        '''
        try:    
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path,exist_ok=True)

            logger.info(f"Extracting the zip file:{self.config.local_data_file}")

            with zipfile.ZipFile(self.config.local_data_file,"r") as zip_ref:
                zip_ref.extractall(unzip_path)
            
            logger.info(f"Extraction completed at: {unzip_path}")
        
        except Exception as e:
            logger.exception("Failed during extracting the zip file")
            raise e
        
    def split_data(self, train_ratio=0.7, val_ratio=0.15):
        
        source_dir = self.config.dataset_dir

        train_dir = self.config.train_dir
        val_dir = self.config.val_dir
        test_dir = self.config.test_dir

        if train_dir.exists() and val_dir.exists() and test_dir.exists():
            logger.info("Train/Val/Test directories already exist. Skipping split.")
            return

        logger.info("Creating Train/Val/Test split")

        assert train_ratio + val_ratio < 1.0, "Train + Val ratio must be < 1"

        random.seed(42)

        split_dirs = {"train", "val", "test"}

        for class_dir in source_dir.iterdir():
            if not class_dir.is_dir():
                continue

            if class_dir.name in split_dirs:
                continue

            images = [p for p in class_dir.iterdir() if p.is_file()]
            random.shuffle(images)

            n_total = len(images)
            n_train = int(train_ratio * n_total)
            n_val = int(val_ratio * n_total)

            splits = {
                "train": images[:n_train],
                "val": images[n_train:n_train + n_val],
                "test": images[n_train + n_val:]
            }

            for split, files in splits.items():
                target_dir = source_dir / split / class_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)

                for img in files:
                    shutil.copy2(img, target_dir / img.name)

        logger.info("Train/Val/Test split created successfully")
