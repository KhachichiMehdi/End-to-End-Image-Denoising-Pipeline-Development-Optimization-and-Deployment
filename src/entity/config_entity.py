from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration class for data ingestion.

    Attributes:
        images_dir (list()): List of paths to directories containing images.
        root_dir (Path): The root directory for storing data.
        train_data_path (Path): Path to store the training data.
        test_data_path (Path): Path to store the testing data.
        im_size (tuple): The size of the images to be processed (height, width).
        test_split (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """
    images_dir: list
    root_dir : Path
    train_data_path: Path
    test_data_path: Path
    im_size: tuple
    test_split: float
    random_state: int


@dataclass(frozen=True)
class DataPreprocessingConfig:
    """
    Configuration class for data preprocessing.

    Attributes:
        root_dir (Path): The root directory for storing processed data.
        train_data (np.ndarray): The training data as a numpy array.
        test_data (np.ndarray): The testing data as a numpy array.
        x_train_noisy_path (Path): Path to store noisy versions of the training data.
        x_test_noisy_path (Path): Path to store noisy versions of the testing data.
        noise_factor (int): Factor by which noise is added to the data.
    """
    root_dir : Path
    train_data_path : Path
    test_data_path : Path
    x_train_noisy_path: Path
    x_test_noisy_path: Path
    noise_factor : int


@dataclass(frozen=True)
class BaseModelConfig:
    """
    Configuration class for setting up the base autoencoder model.

    Attributes:
        root_dir (Path): The root directory for storing model files.
        base_model_path (Path): Path to save the initial base model.
        updated_base_model_path (Path): Path to save the updated base model after training.
        base_learning_rate (int): The initial learning rate for model training.
        im_size (tuple): The size of the input images for the model (height, width).
    """
    root_dir: Path
    base_model_path: Path
    updated_base_model_path : Path
    base_learning_rate : float
    input_shape: tuple


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration class for model training.

    Attributes:
        root_dir (Path): The root directory for storing trained models and logs.
        train_model_path (Path): Path to save the trained model.
        updated_model_base_path (Path): Path to load the updated base model.
        train_data (np.ndarray): The training data as a numpy array.
        test_data (np.ndarray): The testing data as a numpy array.
        x_train_noisy (np.ndarray): The noisy training data.
        x_test_noisy (np.ndarray): The noisy testing data.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): The batch size used during training.
    """
    root_dir: Path
    train_model_path : Path
    updated_model_base_path :Path
    train_data : np.ndarray
    test_data : np.ndarray
    x_train_noisy : np.ndarray
    x_test_noisy : np.ndarray
    num_epochs : int
    batch_size: int
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    path_of_model: Path
    evaluation_report_path:Path
    X_test : np.ndarray
    x_test_noisy : np.array

    






  


    

