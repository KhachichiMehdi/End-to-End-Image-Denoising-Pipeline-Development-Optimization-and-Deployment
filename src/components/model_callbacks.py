import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from dataclasses import dataclass
@dataclass
class ModelCallback:
    def __init__(self) -> None:
        self.config = config
    def _create_early_stopping_callback(self):
        return EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True, verbose=1)
        
    def _create_reduce_lr_callback(self):
        return ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=50, verbose=1)

    def _get_callbacks(self):
        return [
            self._create_early_stopping_callback,
            self._create_reduce_lr_callback
        ]
