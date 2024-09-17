from ensure import ensure_annotations
import cv2
import yaml
import sys
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
from box.exceptions import BoxValueError
import yaml
from box import ConfigBox
from src.utils.logger import logging
from src.utils.exception import CustomException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns the content as a ConfigBox.

    Args:
        path_to_yaml (Path): path-like input.

    Raises:
        ValueError: if yaml file is empty.
        CustomException: If any other error occurs during loading.

    Returns:
        ConfigBox: Loaded content as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if not content:
                raise BoxValueError("YAML file is empty")
            logging.info(f"YAML file loaded successfully from {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError as e:
        logging.error(f"YAML file is empty: {path_to_yaml}")
        raise CustomException(e, sys) from e
    except Exception as e:
        logging.error(f"Error occurred while loading YAML file: {path_to_yaml}")
        raise CustomException(e, sys) from e


@ensure_annotations
def create_directories(paths: list, verbose=True):
    """
    Creates directories if they do not exist.

    Args:
        paths (list): List of directory paths to create.
        verbose (bool): If True, logs the creation or existence of directories.
    """
    try:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                if verbose:
                    logging.info(f"Directory created: {path}")
            else:
                if verbose:
                    logging.info(f"Directory already exists: {path}")
    except Exception as e:
        logging.error(f"Error occurred while creating directories: {e}")
        raise CustomException(e, sys)




@ensure_annotations
def read_data(path_list: list, im_size: tuple) -> tuple:
    
    try:
        X = []
        y = []

        # Extract the file-names of the datasets we read and create a label dictionary.
        tag2idx = {tag.split(os.path.sep)[-1]: i for i, tag in enumerate(path_list)}
        logging.info(f"Label dictionary created: {tag2idx}")

        for path in path_list:
            for im_file in tqdm(glob(path + "*/*")):  # Read all files in path
                try:
                    # os.path.sep is OS agnostic (either '/' or '\'),[-2] to grab folder name.
                    label = im_file.split(os.path.sep)[-2]
                    im = cv2.imread(im_file, cv2.IMREAD_COLOR)
                    if im is None:
                        logging.warning(f"Failed to read image: {im_file}")
                        continue
                    # By default OpenCV reads with BGR format, convert back to RGB.
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    # Resize to appropriate dimensions. You can try different interpolation methods.
                    im = cv2.resize(im, im_size, interpolation=cv2.INTER_AREA)
                    X.append(im)
                    y.append(tag2idx[label])  # Append the label name to y
                except Exception as e:
                    # In case annotations or metadata are found
                    logging.error(f"Error processing file {im_file}: {e}")

        X = np.array(X)  # Convert list to numpy array.
        y = np.eye(len(np.unique(y)))[y].astype(np.uint8)

        return X, y, tag2idx
    except Exception as e:
        logging.error(f"An error occurred while reading images from directories: {e}")
        raise CustomException(e, sys)




@ensure_annotations
def read_numpy_file(file_path: Path) -> np.ndarray:
    """
    Reads a .npy file and returns the numpy array.

    Args:
        file_path (Path): Path to the .npy file.

    Returns:
        np.ndarray: Data loaded from the .npy file.

    Raises:
        CustomException: If any error occurs during file loading.
    """
    try:
        data = np.load(file_path) 
        logging.info(f"File loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"An error occurred while loading the file: {file_path}")
        raise CustomException(e, sys)

@ensure_annotations
def load_model(path: Path ) -> tf.keras.Model :
                    """
                    Charge le modèle à partir du chemin spécifié dans la configuration.
                    """
                    try:
                        logging.info(f"Loading model from {path}")
                        model = tf.keras.models.load_model(path)
                        logging.info("Model loaded successfully.")
                        return model
                    except Exception as e:
                        logging.error(f"Failed to load the model: {e}")
                        raise CustomException(e, sys)
