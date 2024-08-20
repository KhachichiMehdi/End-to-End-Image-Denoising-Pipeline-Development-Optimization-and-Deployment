from ensure import ensure_annotations
import cv2
import yaml
import sys
import os
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

    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        CustomException: If any other error occurs during loading.

    Returns:
        ConfigBox: Loaded content as a ConfigBox object

    """


    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if not content:
                raise BoxValueError("YAML file is empty")
            logging.info(f"YAML file loaded successfully from {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        logging.error(f"YAML file is empty: {path_to_yaml}")
        raise CustomException(e, sys)
    except Exception as e:
        logging.error(f"Error occurred while loading YAML file: {path_to_yaml}")
        CustomException(e, sys)

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
def read_images_from_directory(path_list: list, im_size: tuple) -> tuple:
    """
    Reads images from directories, processes them, and returns the images, labels, and label dictionary.

    Args:
        path_list (list): List of paths to directories containing images.
        im_size (tuple): Desired image size (width, height).

    Returns:
        tuple: Tuple containing:
            - X (np.ndarray): Array of processed images.
            - y (np.ndarray): Array of one-hot encoded labels.
            - tag2idx (dict): Dictionary mapping label names to indices.

    Raises:
        CustomException: If any error occurs during the image processing.
    """
    try:
        X = []
        y = []
        tag2idx = {tag.split(os.path.sep)[-1]: i for i, tag in enumerate(path_list)}
        logging.info(f"Label dictionary created: {tag2idx}")

        for path in path_list:
            for im_file in tqdm(glob(os.path.join(path, "*/*"))):
                try:
                    label = im_file.split(os.path.sep)[-2]
                    im = cv2.imread(im_file, cv2.IMREAD_COLOR)
                    if im is None:
                        logging.warning(f"Failed to read image: {im_file}")
                        continue

                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, im_size, interpolation=cv2.INTER_AREA)

                    X.append(im)
                    y.append(tag2idx[label])
                except Exception as e:
                    logging.error(f"Error processing file {im_file}: {e}")

        X = np.array(X)
        y = np.eye(len(np.unique(y)))[y].astype(np.uint8)

        logging.info(f"Loaded data shape: {X.shape}")
        logging.info(f"Labels shape: {y.shape}")

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
    
    