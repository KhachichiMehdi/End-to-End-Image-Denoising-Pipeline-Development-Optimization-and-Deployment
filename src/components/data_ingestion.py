import os
import sys
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataIngestionConfig
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.common import read_data 
from sklearn.utils import shuffle



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

    config: DataIngestionConfig

    def save_data(self, path: Path, data: np.ndarray, data_desc: str) -> None:
        """
        Save numpy array data to the specified path.

        Args:
            path (Path): Path to save the numpy file.
            data (np.ndarray): Data to be saved.
            data_desc (str): Description of the data being saved.

        Raises:
            CustomException: If the file cannot be saved due to permission errors.
        """
        try:
            np.save(path, data)
            logging.info(f"{data_desc} saved successfully at {path}")
        except Exception as e:
            logging.error(f"An error occurred while saving {data_desc}: {e}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> None:
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
            images, labels, tag2idx = read_data(self.config.images_dir, self.config.im_size)
            logging.info('Images successfully read from the directory.')

            # Step 2: Split the data into training and testing sets
            train_images, test_images, train_labels, test_labels = train_test_split(
                images, labels, test_size=self.config.test_split,shuffle=True, stratify=labels, random_state=self.config.random_state
            )
            logging.info("Data successfully split into training and testing sets.")

            # Step 3: Save the training and testing data
            self.save_data(self.config.train_data_path, train_images, "training image data")
            self.save_data(self.config.test_data_path, test_images, "testing image data")

            logging.info("Ingestion of the data is completed")

        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise CustomException(e, sys)