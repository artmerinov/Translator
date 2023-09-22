import os
import yaml
import torch


def set_random_seed(seed: int):
    """
    Sets the seed for all devices (both CPU and CUDA).
    """
    torch.manual_seed(seed)


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            setattr(self, key.upper(), value)


def get_weights_file_path(config, epoch: str):
    return os.path.join(f"{config.MODEL_FOLDER_NAME}", f"{epoch}.pt")


def get_latest_weights_file_path(config):
    weights_files = os.listdir(config.MODEL_FOLDER_NAME)
    weights_files.sort()
    return os.path.join(config.MODEL_FOLDER_NAME, str(weights_files[-1]))
