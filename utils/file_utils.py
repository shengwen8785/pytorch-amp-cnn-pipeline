import os
import yaml


def load_yaml(file_path: str):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def create_dirs(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)