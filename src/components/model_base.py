import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from src.entity.config_entity import BaseModelConfig
from src.utils.exception import CustomException
from src.utils.logger import logging  # Import logging
from dataclasses import dataclass
import sys  # Import sys

@dataclass
class BaseModel:
    """
    Class for building and managing the base autoencoder model.

    This class handles the creation, compilation, and saving of an autoencoder model 
    that can be used as the base model for further training or fine-tuning.

    Attributes:
        config (BaseModelConfig): Configuration for the base model preparation process.
        model (tf.keras.Model): The autoencoder model instance.
    """
    
    def __init__(self, config: BaseModelConfig) -> None:
        """
        Initialize the BaseModel class.

        Args:
            config (BaseModelConfig): Configuration for the base model, including parameters 
            like image size, learning rate, and file paths for saving models.
        """
        self.config = config
        self.model = None
        
    def build_autoencoder(self) -> tf.keras.Model:
        """
        Build the autoencoder model architecture.

        This method defines and compiles an autoencoder model with a convolutional
        encoder-decoder architecture. The model is designed to denoise images.

        Returns:
            tf.keras.Model: Compiled autoencoder model.

        Raises:
            CustomException: If any errors occur during the model building process.
        """
        try:
            logging.info("Starting to build the autoencoder model.")
            input_img = layers.Input(shape=self.config.im_size)

            # Encoder
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(input_img)
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)

            # Decoder
            x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2)(x)
            x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2)(x)
            x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2)(x)
            decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

            autoencoder = Model(input_img, decoded)
            autoencoder.compile(optimizer=Adam(learning_rate=self.config.base_learning_rate), loss='mean_squared_error')
            logging.info("Autoencoder model built and compiled successfully.")
            return autoencoder

        except Exception as e:
            logging.error(f"Error occurred while building the autoencoder: {e}")
            raise CustomException(e, sys)

    def get_base_model(self):
        """
        Builds the autoencoder model and saves it to the specified path.

        Raises:
            CustomException: If any errors occur while getting the base model.
        """
        try:
            logging.info("Getting the base model.")
            self.model = self.build_autoencoder()
            self.save_model(path=self.config.base_model_path, model=self.model)
            logging.info(f"Base model saved at {self.config.base_model_path}.")
        except Exception as e:
            logging.error(f"Error occurred while getting the base model: {e}")
            raise CustomException(e, sys)

    def update_base_model(self):
        """
        Updates and saves the base model.

        Raises:
            CustomException: If any errors occur while updating the base model.
        """
        try:
            logging.info("Updating the base model.")
            self.full_model = self.model
            self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
            logging.info(f"Updated base model saved at {self.config.updated_base_model_path}.")
        except Exception as e:
            logging.error(f"Error occurred while updating the base model: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the given Keras model to the specified path.

        Args:
            path (Path): The path where the model will be saved.
            model (tf.keras.Model): The Keras model to be saved.

        Raises:
            CustomException: If any errors occur during the model saving process.
        """
        try:
            logging.info(f"Saving model to {path}.")
            model.save(path)
            logging.info(f"Model saved successfully at {path}.")
        except Exception as e:
            logging.error(f"Error occurred while saving the model: {e}")
            raise CustomException(e, sys)
