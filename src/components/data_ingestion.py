import warnings

warnings.filterwarnings("ignore")

import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob
from pathlib import Path
import sys 
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

@dataclass

class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.npy")
    test_data_path: str = os.path.join('artifacts', "test.npy")
    raw_data_path: str = os.path.join('artifacts', "raw.npy")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
     

     def read_images_from_directory(self,path,im_size):
          X=[]
          y=[]
          tag2idix={tag.split(os.path.sep)[-1]: i for i, tag in enumerate(path_list)}
          logging.info("Label dictionary created: %s", tag2idx)


     def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Ensure directories exist
            os.makedirs(self.ingestion_config.raw_data_path, exist_ok=True)
            os.makedirs(self.ingestion_config.train_data_path, exist_ok=True)
            os.makedirs(self.ingestion_config.test_data_path, exist_ok=True)
            
            # Read images
            image_dir = 'path_to_your_images_directory'
            images = self.read_images_from_directory(image_dir)
            
            # Save raw data
            np.save(os.path.join(self.ingestion_config.raw_data_path, 'data.npy'), images)
            logging.info('Saved raw images data')

            # Split data
            train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
            
            # Save train and test data
            np.save(os.path.join(self.ingestion_config.train_data_path, 'train.npy'), train_images)
            np.save(os.path.join(self.ingestion_config.test_data_path, 'test.npy'), test_images)
            logging.info('Saved train and test images data')

            logging.info("Ingestion of the data is completed")

            return (os.path.join(self.ingestion_config.train_data_path, 'train.npy'),
                    os.path.join(self.ingestion_config.test_data_path, 'test.npy'))

        except Exception as e:
            raise CustomException(e, sys)
          









