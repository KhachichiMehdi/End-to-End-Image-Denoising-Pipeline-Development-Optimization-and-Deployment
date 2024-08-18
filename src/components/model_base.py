import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from src.entity.config_entity import BaseModelConfig
from dataclasses import dataclass

@dataclass
class PrepareBaseModel:
    def __init__(self, config: BaseModelConfig) -> None:
        self.config = config
        self.model = None

    def build_autoencoder(self):
        try:
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
            return autoencoder
        except Exception as e:
            print(f"Error occurred while building the autoencoder: {e}")
            raise

    def get_base_model(self):
        try:
            self.model = self.build_autoencoder()
            self.save_model(path=self.config.base_model_path, model=self.model)
        except Exception as e:
            print(f"Error occurred while getting the base model: {e}")
            raise

    def update_base_model(self):
        try:
            self.full_model = self.model
            self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        except Exception as e:
            print(f"Error occurred while updating the base model: {e}")
            raise

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        try:
            model.save(path)
        except Exception as e:
            print(f"Error occurred while saving the model: {e}")
            raise
