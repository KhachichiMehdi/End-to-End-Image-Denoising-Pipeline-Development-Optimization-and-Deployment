from dataclasses import dataclass
import numpy as np
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    images_dir: list(Path)
    root_dir : Path
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path
    im_size: tuple
    test_split: float
    random_state: int
@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir : Path
    train_data : np.ndarray
    test_data : np.ndarray
    x_train_noisy_path: Path
    x_test_noisy_path: Path
    noise_factor : int
@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path : Path
    base_learning_rate : int
    im_size : int
  


    

