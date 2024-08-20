import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging
import sys

@dataclass
class ModelCallback:
    """
    Class for creating and managing callbacks used during model training.

    This class handles the creation of Keras callbacks such as EarlyStopping and
    ReduceLROnPlateau, which can be used to optimize the training process by preventing 
    overfitting and adjusting the learning rate when necessary.
    """

    def __init__(self) -> None:
        """
        Initialize the ModelCallback class.
        """
        # No specific configuration is needed for this class as it uses hardcoded values.
        pass

    def _create_early_stopping_callback(self) -> EarlyStopping:
        """
        Create an EarlyStopping callback.

        The EarlyStopping callback is used to stop training when the validation loss
        has stopped improving for a certain number of epochs.

        Returns:
            EarlyStopping: Configured EarlyStopping callback instance.

        Raises:
            CustomException: If an error occurs while creating the EarlyStopping callback.
        """
        try:
            logging.info("Creating EarlyStopping callback.")
            return EarlyStopping(
                monitor="val_loss", 
                patience=100, 
                restore_best_weights=True, 
                verbose=1
            )
        except Exception as e:
            logging.error(f"Error occurred while creating EarlyStopping callback: {e}")
            raise CustomException(e, sys)

    def _create_reduce_lr_callback(self) -> ReduceLROnPlateau:
        """
        Create a ReduceLROnPlateau callback.

        The ReduceLROnPlateau callback is used to reduce the learning rate when the validation loss
        has stopped improving for a certain number of epochs.

        Returns:
            ReduceLROnPlateau: Configured ReduceLROnPlateau callback instance.

        Raises:
            CustomException: If an error occurs while creating the ReduceLROnPlateau callback.
        """
        try:
            logging.info("Creating ReduceLROnPlateau callback.")
            return ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.8, 
                patience=50, 
                verbose=1
            )
        except Exception as e:
            logging.error(f"Error occurred while creating ReduceLROnPlateau callback: {e}")
            raise CustomException(e, sys)

    def _get_callbacks(self) -> list:
        """
        Get a list of configured callbacks to be used during model training.

        This method combines the EarlyStopping and ReduceLROnPlateau callbacks
        into a single list that can be passed to the training function.

        Returns:
            list: List containing the EarlyStopping and ReduceLROnPlateau callbacks.

        Raises:
            CustomException: If an error occurs while getting the callbacks.
        """
        try:
            logging.info("Getting list of callbacks.")
            callbacks = [
                self._create_early_stopping_callback(),
                self._create_reduce_lr_callback()
            ]
            logging.info("Callbacks created successfully.")
            return callbacks
        except Exception as e:
            logging.error(f"Error occurred while getting callbacks: {e}")
            raise CustomException(e, sys)
