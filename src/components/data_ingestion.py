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
    """
    Class for handling the data ingestion process.

    This class is responsible for reading image data from directories, 
    splitting it into training and testing datasets, and saving the 
    processed data to specified paths.

    Attributes:
        config (DataIngestionConfig): Configuration for the data ingestion process.
    """


    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def initiate_data_ingestion(self)  -> None :
        """
        Executes the data ingestion process.

        This method reads image data from the specified directory, splits it into
        training and testing datasets, and saves the resulting datasets to disk.

        Raises:
            CustomException: If any errors occur during the data ingestion process.
        """

        logging.info("Entered the data ingestion method or component")
    
        
        try:
            # Step 1: Read images from the directory
            logging.info("Reading Images from directory")
            images, labels, tag2idx = read_images_from_directory(self.config.images_dir,self.config.im_size)
            logging.info('Images successfully read from the directory.')


            

            # Step 2: Save the raw images data
            np.save(self.config.raw_data_path, images)
            logging.info('Saved raw images data')

            # Step 3: Split the data into training and testing sets
            train_images, test_images, train_labels, test_labels = train_test_split(
                images, labels, test_size=self.config.test_split, stratify= labels, random_state=self.config.random_state )
            logging.info("Data successfully split into training and testing sets.")


            # Step 4: Save the training and testing dataa
            np.save(self.config.train_data_path, train_images)
            np.save(self.config.test_data_path, test_images)
            logging.info(f"Training image data saved at {self.config.train_data_path}")
            logging.info(f"Testing image data saved at {self.config.test_data_path}")

            logging.info("Ingestion of the data is completed")


        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise CustomException(e, sys)
