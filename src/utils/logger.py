import logging.config
import os
import sys
import yaml
from typing import Optional, Dict, Any

def setup_logging(
    config_path: str = "config/logging.yaml",
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG"
) -> None:
    """
    Setup logging configuration from a YAML file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the logging configuration file, by default "config/logging.yaml"
    default_level : int, optional
        Default logging level if configuration file is not found, by default logging.INFO
    env_key : str, optional
        Environment variable that can be used to override the config path, by default "LOG_CFG"
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Check if the config path was overridden by an environment variable
    config_path_env = os.getenv(env_key, None)
    if config_path_env:
        config_path = config_path_env
    
    # Load configuration if the file exists
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error loading logging configuration from {config_path}: {e}")
                print("Using default configuration.")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print(f"Logging config file not found at {config_path}. Using default configuration.")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Parameters
    ----------
    name : str
        Name of the logger
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
