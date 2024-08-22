import sys
from pathlib import Path
import numpy as np
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.common import create_directories, read_yaml ,read_numpy_file
from src.entity.config_entity import DataIngestionConfig , DataPreprocessingConfig ,BaseModelConfig ,TrainingConfig

class Configuration:
    def __init__(self, config_file_path: Path, params_file_path: Path):
        try:
            # Load configurations
            self.config = read_yaml(config_file_path)
            self.params = read_yaml(params_file_path)
            
            # Create necessary directories
            self._create_artifacts_directory()
    

        except Exception as e:
            raise CustomException(e, sys)


    def _create_artifacts_directory(self):
        """ CREATE THE ROOT ARIFICATS DIRECTORY IF IT DOES NOT EXIST """
        artifacts_root = self.config.artifacts_root
        if not Path(artifacts_root).exists:
             create_directories([artifacts_root])
        else:
            logging.info(f"Artifacts root directory already exists: {artifacts_root}")


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            images_dir=list(config.images_dir),
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            im_size=tuple(list(self.params.im_size)),
            test_split=self.params.test_split,
            random_state=self.params.random_state
        )
        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        create_directories([config.root_dir])
        
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            train_data=read_numpy_file(Path(self.get_data_ingestion_config().train_data_path)),
            test_data=read_numpy_file(Path(self.get_data_ingestion_config().test_data_path)),
            x_train_noisy_path=config.x_train_noisy_path,
            x_test_noisy_path=config.x_test_noisy_path,
            noise_factor=self.params.noise_factor
        )
        return data_preprocessing_config

    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model
        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=config.base_model_path,
            updated_base_model_path=config.updated_base_model_path,
            base_learning_rate=float(self.params.base_learning_rate),
            input_shape=tuple(list(self.params.input_shape))
        )
        return base_model_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        create_directories([config.root_dir])
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir), #data
            train_model_path=Path(training.train_model_path), # artifacts/training/
            updated_model_base_path=self.get_base_model_config().updated_base_model_path,
            train_data = read_numpy_file(self.get_data_ingestion_config().train_data_path),
            test_data = read_numpy_file(self.get_data_ingestion_config().test_data_path),
            x_train_noisy =read_numpy_file(self.get_data_preprocessing_config().x_train_noisy_path),
            x_test_noisy = read_numpy_file(self.get_data_preprocessing_config().x_test_noisy_path),
            num_epochs = self.params.num_epochs,
            batch_size = self.params.batch_size
        )
        return training_config





    





