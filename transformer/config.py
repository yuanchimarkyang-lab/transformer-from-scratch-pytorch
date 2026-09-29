"""
Load the configuration files (in configs/)

This script defines a function to load the configurations defined in the configuration files.

"""


from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(config_path="configs/env.yaml"):
    """
    Load a YAML config file relative to the project root
    """

    full_path = PROJECT_ROOT / config_path

    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config
