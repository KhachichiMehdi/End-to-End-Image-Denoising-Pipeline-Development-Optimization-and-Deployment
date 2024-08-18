import warnings
warnings.filterwarnings("ignore")
import numpy as np
import sys
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.entity.config_entity import DataIngestionConfig 
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.common import read_images_from_directory
from dataclasses import dataclass


@dataclass
class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def initiate_data_ingestion(self)  -> None :
        logging.info("Entered the data ingestion method or component")
        
        try:
            # Read images
            logging.info("Reading Images from directory")
            images, labels, tag2idx = read_images_from_directory(self.config.images_dir,self.config.im_size)
            logging.info('Images read from path.')


            

            # Save raw data
            np.save(self.config.raw_data_path, images)
            logging.info('Saved raw images data')

            # Split data
            train_images, test_images, train_labels, test_labels = train_test_split(
                images, labels, test_size=self.config.test_split, stratify= labels, random_state=self.config.random_state )

            # Save train and test data
            np.save(self.config.train_data_path, train_images)
            np.save(self.config.test_data_path, test_images)
            logging.info('Saved train and test images data')

            logging.info("Ingestion of the data is completed")


        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise CustomException(e, sys)
