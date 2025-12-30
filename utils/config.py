"""
Configuration utility for loading model arguments
"""
import json


class EasyDict(dict):
    """Dictionary that allows attribute access"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return EasyDict(config_dict)
