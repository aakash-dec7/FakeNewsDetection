import os
import re
import yaml
import json
from pathlib import Path
from typing import Union
from box import ConfigBox
from src.fnd.logger import logger
from ensure import ensure_annotations


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox for dot notation access."""
    if not path_to_yaml.exists():
        logger.error(f"YAML file not found: {path_to_yaml.resolve()}")
        raise FileNotFoundError(f"YAML file not found: {path_to_yaml.resolve()}")

    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            data = yaml.safe_load(yaml_file) or {}
            return ConfigBox(data)
    except Exception as e:
        logger.exception(f"Error reading YAML file {path_to_yaml.resolve()}: {str(e)}")
        raise RuntimeError(
            f"Error reading YAML file {path_to_yaml.resolve()}: {str(e)}"
        )


@ensure_annotations
def update_yaml_file(
    section: str, key: str, value: Union[str, int, float, bool, dict, list], path: str
):
    """Updates a nested key in the YAML file at the given path."""
    try:
        try:
            with open(path, "r") as file:
                data = yaml.safe_load(file) or {}
        except FileNotFoundError:
            data = {}

        if section not in data:
            data[section] = {}

        data[section][key] = value

        with open(path, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

        logger.debug(f"YAML file '{path}' updated successfully.")
    except Exception as e:
        logger.exception(f"Error updating YAML file: {e}")


@ensure_annotations
def create_directories(path):
    """Creates directories if they don't exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.exception(f"Error creating directories: {path}")
        raise e


@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """Saves data as a JSON file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            logger.debug(f"JSON file successfully saved: {path}")
    except (TypeError, Exception) as e:
        logger.exception(f"Error saving JSON file: {path}")
        raise e


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads data from a JSON file and returns it as a ConfigBox."""
    if not path.exists():
        logger.error(f"JSON file not found: {path}")
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug(f"JSON file successfully loaded: {path}")
            return ConfigBox(data)
    except (json.JSONDecodeError, Exception) as e:
        logger.exception(f"Error loading JSON file: {path}")
        raise e


@ensure_annotations
def get_package_info():
    """Retrieves package name and version from setup.py."""
    try:
        with open("setup.py", "r") as f:
            setup_content = f.read()
    except FileNotFoundError:
        return "unknown", "unknown"

    name_match = re.search(r"name=['\"]([^'\"]+)['\"]", setup_content)
    version_match = re.search(r"version=['\"]([^'\"]+)['\"]", setup_content)

    project_name = name_match.group(1) if name_match else "unknown"
    version = version_match.group(1) if version_match else "unknown"

    return project_name, version
