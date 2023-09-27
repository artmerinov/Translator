import os
import yaml
import torch
import numpy as np


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            setattr(self, key.upper(), value)


def set_random_seed(seed: int):
    """
    Fixes random state for reproducibility. 
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_last_checkpoint_path(config):
    """
    Loads last model weights.
    """
    weights_files = os.listdir(config.MODEL_FOLDER_NAME)
    weights_files.sort()
    return os.path.join(config.MODEL_FOLDER_NAME, str(weights_files[-1]))