import os
import pandas as pd
from src.fnd.logger import logger
from src.fnd.entity.entity import DataValidationConfig
from src.fnd.config.configuration import ConfigurationManager


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        """Initialize DataValidation with the provided configuration."""
        self.config = config
        self.expected_columns = self.config.schema.num_columns

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load the dataset from the given path."""
        try:
            logger.info(f"Loading dataset from: {file_path}")
            data = pd.read_csv(file_path, header=None)
            logger.info(
                f"Dataset loaded successfully from {file_path} with shape: {data.shape}"
            )
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            logger.error(f"The file at {file_path} is empty.")
        except pd.errors.ParserError:
            logger.error(f"Error parsing {file_path}. Invalid format.")
        except Exception as e:
            logger.error(
                f"Unexpected error while loading data from {file_path}: {repr(e)}"
            )
        return None

    def _validate_columns(self, data: pd.DataFrame, file_path: str) -> bool:
        """Check if the dataset has the expected number of columns."""
        if data is None:
            logger.error(
                f"Skipping column validation for {file_path} as data is not loaded."
            )
            return False

        if data.shape[1] != self.expected_columns:
            logger.error(
                f"Column mismatch in {file_path}. Expected {self.expected_columns} columns, but found {data.shape[1]}"
            )
            return False

        logger.info(f"Column validation passed for {file_path}.")
        return True

    def run(self) -> None:
        """Run the complete validation pipeline for train, test, and validation datasets."""
        for file_path in [
            self.config.train_data_path,
            self.config.test_data_path,
            self.config.valid_data_path,
        ]:
            data = self._load_data(file_path)
            if data is None or not self._validate_columns(data, file_path):
                logger.error(f"Validation failed for {file_path}. Stopping pipeline.")
                raise RuntimeError("Data validation failed.")
            logger.info(f"Validation successful for {file_path}.")


if __name__ == "__main__":
    try:
        data_validation_config = ConfigurationManager().get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.run()
    except Exception as e:
        logger.exception("Data validation pipeline failed.")
        raise RuntimeError("Data validation pipeline failed.") from e
