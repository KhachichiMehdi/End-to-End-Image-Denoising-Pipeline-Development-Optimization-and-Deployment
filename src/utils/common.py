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
        e: empty file

    Returns:
        ConfigBox: ConfigBox type

    """


    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
          logging.error(f"Error loading YAML file: {path_to_yaml}")
          CustomException(e, sys)

@ensure_annotations
def create_directories(paths: list, verbose=True):
    """Create directories if they do not exist."""
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            if verbose:
                logging.info(f"Directory created: {path}")
        else:
            if verbose:
                logging.info(f"Directory already exists: {path}")


@ensure_annotations
def read_images_from_directory(self, path_list:list ,im_size:tuple) -> tuple:
        X = []
        y = []

        # Extract the file-names of the datasets we read and create a label dictionary.
        tag2idx = {tag.split(os.path.sep)[-1]: i for i, tag in enumerate(path_list)}
        logging.info(f'Label dictionary created: {tag2idx}')

        for path in path_list:
            for im_file in tqdm(glob(os.path.join(path, "*/*"))):  # Read all files in path
                try:
                    # os.path.separator is OS agnostic (either '/' or '\'),[-2] to grab folder name.
                    label = im_file.split(os.path.sep)[-2]
                    im = cv2.imread(im_file, cv2.IMREAD_COLOR)
                    if im is None:
                        logging.warning(f'Failed to read image: {im_file}')
                        continue

                    # By default OpenCV reads with BGR format, convert back to RGB.
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                    # Resize to appropriate dimensions. You can try different interpolation methods.
                    im = cv2.resize(im, im_size, interpolation=cv2.INTER_AREA)

                    X.append(im)
                    y.append(tag2idx[label])  # Append the label name to y
                except Exception as e:
                    # Log the exception if an error occurs
                    logging.error(f'Error processing file {im_file}: {e}')

        X = np.array(X)  # Convert list to numpy array.
        y = np.eye(len(np.unique(y)))[y].astype(np.uint8)

        logging.info(f'Loaded data shape: {X.shape}')
        logging.info(f'Labels shape: {y.shape}')

        return X, y, tag2idx
@ensure_annotations
def read_numpy_file(file_path: Path) -> np.ndarray:
    """
    Reads a .npy file and returns the numpy array.

    Args:
        file_path (str): Path to the .npy file.

    Returns:
        np.ndarray: The data loaded from the .npy file.
    """
    try:
        data = np.load(file_path)
        logging.info(f"File loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"An error occurred while loading the file: {e}")
    