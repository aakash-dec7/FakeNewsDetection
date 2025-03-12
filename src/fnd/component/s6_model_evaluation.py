import os
import json
import joblib
import mlflow
import dagshub
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report
from src.fnd.logger import logger
from src.fnd.entity.entity import ModelEvaluationConfig
from src.fnd.config.configuration import ConfigurationManager
from src.fnd.utils.utils import get_package_info


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> None:
        """Initialize the model evaluation pipeline."""
        self.config = config
        self.model = self._load_model()
        self.project_name, self.version = get_package_info()
        self._initialize()

    def _initialize(self) -> None:
        """Initialize MLflow and load validation data."""
        self._init_mlflow()
        self._load_data()

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking with DagsHub."""
        try:
            dagshub.init(
                repo_owner=self.config.repo_owner,
                repo_name=self.config.repo_name,
                mlflow=True,
            )
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment(f"v{self.version}")
            self.run_name = (
                f"v{self.version}--{datetime.now().strftime('%Y/%m/%d-%H:%M:%S')}"
            )
            logger.info("MLflow tracking initialized.")
        except Exception as e:
            logger.exception("Error initializing MLflow: %s", str(e))
            raise

    def _load_data(self) -> None:
        """Load the validation data from the specified paths."""
        try:
            logger.info(
                f"Loading validation data from: {self.config.test_X_path} and {self.config.test_y_path}"
            )
            self.X_valid = np.load(self.config.test_X_path)
            self.y_valid = np.load(self.config.test_y_path)
        except Exception as e:
            logger.exception("Error loading validation data: %s", str(e))
            raise

    def _load_model(self) -> joblib:
        """Load the trained model from the specified path."""
        try:
            logger.info(f"Loading model from: {self.config.model_path}")
            model = joblib.load(self.config.model_path)
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.exception("Failed to load model: %s", str(e))
            raise

    def _evaluate_model(self) -> dict:
        """Evaluate model performance and return classification report."""
        try:
            logger.info("Evaluating model...")
            y_pred = self.model.predict(self.X_valid)
            return classification_report(self.y_valid, y_pred, output_dict=True)
        except Exception as e:
            logger.exception("Error evaluating model: %s", str(e))
            raise

    def _get_git_commit_hash(self) -> str:
        """Retrieve the current Git commit hash for version tracking."""
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
            )
        except subprocess.CalledProcessError:
            logger.warning("Could not retrieve Git commit hash.")
            return "unknown"

    def _log_results(self, report: dict) -> None:
        """Log model evaluation results to MLflow and save metrics."""
        commit_hash = self._get_git_commit_hash()
        logger.info(f"Commit Hash: {commit_hash}")

        try:
            with mlflow.start_run(run_name=self.run_name):
                mlflow.set_tag("mlflow.source.git.commit", commit_hash)
                mlflow.log_metric("accuracy", report.get("accuracy", 0.0))
                mlflow.log_metric(
                    "macro_avg_f1", report.get("macro avg", {}).get("f1-score", 0.0)
                )
                mlflow.log_metric(
                    "weighted_avg_f1",
                    report.get("weighted avg", {}).get("f1-score", 0.0),
                )
                logger.info("Model evaluation logged.")
            mlflow.end_run()
        except Exception as e:
            logger.error("Error during MLflow run: %s", str(e))
            raise

        metrics_file = Path(self.config.metrics_path) / "classification_report.json"
        with open(metrics_file, "w") as f:
            json.dump(report, f, indent=4)
        logger.info("Evaluation complete!")

    def run(self) -> None:
        """Execute the full model evaluation pipeline."""
        report = self._evaluate_model()
        self._log_results(report)


if __name__ == "__main__":
    try:
        model_evaluation_config = ConfigurationManager().get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.run()
    except Exception as e:
        raise RuntimeError("Model evaluation pipeline failed.") from e
