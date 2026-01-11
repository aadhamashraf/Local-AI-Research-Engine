"""
Utilities - Configuration loading and logging setup
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger
import sys
from dotenv import load_dotenv
import os


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment variables
    load_dotenv()
    
    # Override with environment variables if present
    if "OLLAMA_BASE_URL" in os.environ:
        config["ollama"]["base_url"] = os.environ["OLLAMA_BASE_URL"]
    
    if "OLLAMA_LLM_MODEL" in os.environ:
        config["ollama"]["llm_model"] = os.environ["OLLAMA_LLM_MODEL"]
    
    if "OLLAMA_EMBEDDING_MODEL" in os.environ:
        config["ollama"]["embedding_model"] = os.environ["OLLAMA_EMBEDDING_MODEL"]
    
    return config


def setup_logging(config: Dict[str, Any]):
    """
    Setup logging with loguru.
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get("logging", {})
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=log_config.get("level", "INFO"),
        format=log_config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    )
    
    # Add file handler if logs directory specified
    logs_dir = config.get("paths", {}).get("logs")
    if logs_dir:
        logs_path = Path(logs_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            logs_path / "research_engine.log",
            level=log_config.get("level", "INFO"),
            format=log_config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"),
            rotation=log_config.get("rotation", "10 MB")
        )


def ensure_directories(config: Dict[str, Any]):
    """
    Ensure all required directories exist.
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get("paths", {})
    
    for path_key, path_value in paths.items():
        if path_key != "keyword_index" and path_key != "graph_db":  # Skip file paths
            path = Path(path_value)
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
