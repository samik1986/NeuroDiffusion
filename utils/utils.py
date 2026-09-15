import yaml
import os

def load_config(config_path="config.yaml"):
    """Loads parameters from the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    return config
