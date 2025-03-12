from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    root_dir: Path
    download_path: Path


@dataclass
class DataValidationConfig:
    train_data_path: Path
    test_data_path: Path
    valid_data_path: Path
    schema: dict


@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    vectorizer_path: Path


@dataclass
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path


@dataclass
class ModelTrainingConfig:
    root_dir: Path
    train_X_path: Path
    train_y_path: Path


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_X_path: Path
    test_y_path: Path
    model_path: Path
    metrics_path: Path
    repo_name: str
    repo_owner: str
    mlflow_uri: str


@dataclass
class PredictionConfig:
    model_path: Path
    vectorizer_path: Path
