import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import tensorflow as tf
import sys
from src.entity.config_entity import DataPreprocessingConfig 
from src.utils.logger import logging
from src.utils.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataPreprocessing:
    config: DataPreprocessingConfig

    def __post_init__(self):
        self._normalize_data()
        self._add_noise()

    def _normalize_data(self) -> None:
        try:
            logging.info("Normalizing the data by scaling it to the range [0, 1].")
            self.config.train_data = self.config.train_data.astype("float32") / 255.0
            self.config.test_data = self.config.test_data.astype("float32") / 255.0
            logging.info(f"x_train shape: {self.config.train_data.shape}, x_test shape: {self.config.test_data.shape}")
        except Exception as e:
            logging.error(f"An error occurred while normalizing the data: {e}")
            raise CustomException(e, sys)

    def _add_noise(self) -> None:
        try:
            logging.info(f"Adding noise to the data with a noise factor of {self.config.noise_factor}.")
            x_train_noisy = self.config.train_data + self.config.noise_factor * tf.random.normal(shape=self.config.train_data.shape)
            x_test_noisy = self.config.test_data + self.config.noise_factor * tf.random.normal(shape=self.config.test_data.shape)
            
            # Clipping to maintain pixel values in the range [0, 1]
            x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0.0, clip_value_max=1.0)
            x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0.0, clip_value_max=1.0)
            
            logging.info("Noise added and data clipped to the range [0, 1].")
            
            # Save the noisy data
            np.save(self.config.x_train_noisy_path, x_train_noisy)
            np.save(self.config.x_test_noisy_path, x_test_noisy)
            logging.info('Saved noisy train and test data.')
            logging.info('Preprocessing of data is completed')


        except Exception as e:
            logging.error(f"An error occurred while adding noise to the data: {e}")
            raise CustomException(e, sys)
