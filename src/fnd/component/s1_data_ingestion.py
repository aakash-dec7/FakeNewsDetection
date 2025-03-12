import os
import pandas as pd
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from src.fnd.logger import logger
from src.fnd.entity.entity import DataIngestionConfig
from src.fnd.config.configuration import ConfigurationManager


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        """
        Initialize DataIngestion with the provided configuration.
        """
        self.config: DataIngestionConfig = config
        self.dataset_name: str = "doanquanvietnamca/liar-dataset"
        self.kaggle_json_path: Path = Path.home() / ".kaggle"
        self.root_dir: Path = Path(self.config.root_dir)
        self.download_path: Path = Path(self.config.download_path)
        self._set_kaggle_credentials()

    def _set_kaggle_credentials(self) -> None:
        """
        Set Kaggle API credentials.
        """
        try:
            os.environ["KAGGLE_CONFIG_DIR"] = str(self.kaggle_json_path)
            kaggle_json: Path = self.kaggle_json_path / "kaggle.json"

            if not kaggle_json.exists():
                raise FileNotFoundError(
                    "kaggle.json file not found. Please upload your Kaggle API key."
                )

            logger.info("Kaggle API credentials set successfully.")
        except Exception as e:
            logger.exception("Failed to set Kaggle API credentials.")
            raise RuntimeError("Error setting Kaggle credentials") from e

    def _download_dataset(self) -> None:
        """
        Download the dataset if it doesn't already exist.
        """
        if self.download_path.exists() and any(self.download_path.iterdir()):
            logger.info(f"Dataset already exists in: {self.download_path}")
            return

        try:
            logger.info(
                f"Downloading from {self.dataset_name} to {self.download_path}..."
            )
            self.download_path.mkdir(parents=True, exist_ok=True)

            api: KaggleApi = KaggleApi()
            api.authenticate()
            logger.info("Kaggle API authentication successful.")

            api.dataset_download_files(
                self.dataset_name, path=str(self.download_path), unzip=True
            )
            logger.info(f"Download successful. Files saved at: {self.download_path}")
        except Exception as e:
            logger.exception("Dataset download failed.")
            raise RuntimeError("Dataset download failed") from e

    def _convert_tsv_to_csv(self) -> None:
        """
        Convert TSV files to CSV format and delete the original TSV files.
        """
        columns: list[str] = [
            "id",
            "label",
            "statement",
            "subject",
            "speaker",
            "job_title",
            "state_info",
            "party_affiliation",
            "barely_true_counts",
            "false_counts",
            "half_true_counts",
            "mostly_true_counts",
            "pants_on_fire_counts",
            "context",
        ]

        tsv_files: list[Path] = list(self.download_path.glob("*.tsv"))
        if not tsv_files:
            logger.warning(
                "No TSV files found for conversion. Check if dataset format has changed."
            )
            return

        for file in tsv_files:
            csv_path: Path = file.with_suffix(".csv")
            df: pd.DataFrame = pd.read_csv(file, sep="\t", header=None, names=columns)

            if df.shape[1] != len(columns):
                logger.warning(
                    f"File {file} may have incorrect columns. Expected {len(columns)}, got {df.shape[1]}."
                )

            df.to_csv(csv_path, index=False)
            logger.info(f"Converted {file} to {csv_path}.")
            file.unlink()
            logger.info(f"Deleted original TSV file: {file}")

    def run(self) -> None:
        """
        Run the data ingestion pipeline.
        """
        self._download_dataset()
        self._convert_tsv_to_csv()


if __name__ == "__main__":
    try:
        data_ingestion_config: DataIngestionConfig = (
            ConfigurationManager().get_data_ingestion_config()
        )
        data_ingestion: DataIngestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.run()
    except Exception as e:
        logger.exception("Data ingestion pipeline failed.")
        raise RuntimeError("Data ingestion pipeline failed.") from e
