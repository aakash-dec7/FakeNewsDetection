import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.fnd.logger import logger
from src.fnd.entity.entity import DataTransformationConfig
from src.fnd.config.configuration import ConfigurationManager


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """Initialize DataTransformation with configuration settings."""
        self.config = config
        self.train_dir = os.path.join(self.config.root_dir, "train")
        self.test_dir = os.path.join(self.config.root_dir, "test")

    def _load_preprocessed_data(self, file_path: str) -> pd.DataFrame:
        """Load preprocessed CSV file and convert embeddings from strings to numpy arrays."""
        try:
            logger.info(f"Loading preprocessed data from {file_path}")
            df = pd.read_csv(file_path)
            df["statement"] = df["statement"].apply(
                lambda x: np.array(json.loads(x), dtype=np.float32)
            )
            logger.info(f"Successfully loaded data from {file_path}")
            return df
        except Exception as e:
            logger.exception(f"Error loading preprocessed data from {file_path}: {e}")
            raise

    def _save_transformed_data(self, data: pd.DataFrame, data_type: str) -> None:
        """Save transformed data as .npy files."""
        try:
            save_path = os.path.join(self.config.root_dir, data_type)
            os.makedirs(save_path, exist_ok=True)

            X = np.stack(data["statement"].values)
            y = data["label"].values.astype(np.int32)

            np.save(os.path.join(save_path, "X.npy"), X)
            np.save(os.path.join(save_path, "y.npy"), y)

            logger.info(
                f"Successfully saved transformed {data_type} data to {save_path}"
            )
        except Exception as e:
            logger.exception(f"Error saving transformed {data_type} data: {e}")
            raise

    def run(self) -> None:
        """Execute the full data transformation pipeline."""
        try:
            logger.info("Starting data transformation pipeline...")
            train_data = self._load_preprocessed_data(self.config.train_data_path)
            test_data = self._load_preprocessed_data(self.config.test_data_path)

            self._save_transformed_data(train_data, "train")
            self._save_transformed_data(test_data, "test")

            logger.info("Data transformation pipeline completed successfully.")
        except Exception as e:
            logger.exception("Data transformation pipeline failed.")
            raise RuntimeError("Data transformation pipeline failed.") from e


if __name__ == "__main__":
    try:
        data_transformation_config = (
            ConfigurationManager().get_data_transformation_config()
        )
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.run()
    except Exception as e:
        logger.exception("Fatal error in data transformation pipeline.")
        raise RuntimeError("Data transformation pipeline terminated.") from e
