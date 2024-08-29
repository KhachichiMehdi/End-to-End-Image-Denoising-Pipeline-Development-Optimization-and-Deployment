import tensorflow as tf
from pathlib import Path
from ..entity.config_entity import TrainingConfig
from dataclasses import dataclass
from src.utils.exception import CustomException
from ..utils.logger import logging
import sys
from sklearn.utils import shuffle

@dataclass
class ModelTraining:
    """
    Class for managing the training process of the autoencoder model.

    This class handles loading the base model, training it with noisy and clean image data, 
    and saving the trained model to a specified path.

    Attributes:
        config (TrainingConfig): Configuration for the training process, 
        including paths, hyperparameters, and data.
        model (tf.keras.Model): The autoencoder model instance to be trained.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialize the Training class with a configuration file.

        Args:
            configfile (TrainingConfig): Configuration for the training process, 
            including paths to data, model, and training parameters.
        """
        self.config = config
        self.model = None

    def get_base_model(self) -> None:
        """
        Load the base autoencoder model from the specified path.

        This method loads the pre-trained or initial autoencoder model that 
        will be used as the starting point for further training.

        Raises:
            Exception: If the model cannot be loaded.
        """
        try:
            self.model = tf.keras.models.load_model(self.config.updated_model_base_path)
            logging.info(f"Loaded base model from {self.config.updated_model_base_path}.")
        except Exception as e:
            logging.error(f"Error occurred while loading the base model: {e}")
            raise CustomException(e, sys)



    def train(self, callbacks_list: list) -> None:
        """
        Train the autoencoder model using noisy and clean image data.

        This method trains the model using the training data provided in the configuration,
        with the ability to monitor training progress and adjust the training process using callbacks.

        Args:
            callbacks_list (list): List of Keras callbacks to be used during training.

        Raises:
            Exception: If an error occurs during training.
        """
        try:
            logging.info("Starting the training process.")
            self.model.fit(
                self.config.x_train_noisy, self.config.train_data,
                epochs=self.config.num_epochs,
                batch_size=self.config.batch_size,
                shuffle=True,
                validation_data=(self.config.x_test_noisy, self.config.test_data),
                callbacks=callbacks_list,
                verbose=1
            )
            logging.info("Training completed successfully.")
            
            self.save_model(path=self.config.train_model_path, model=self.model)
            logging.info(f"Trained model saved at {self.config.train_model_path}.")
        except Exception as e:
            logging.error(f"Error occurred during training: {e}")
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
