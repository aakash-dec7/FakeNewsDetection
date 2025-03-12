import os
import json
import joblib
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.fnd.logger import logger
from src.fnd.entity.entity import ModelTrainingConfig
from src.fnd.config.configuration import ConfigurationManager


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        """Initialize model training with configuration parameters."""
        self.config = config

        # Load training data
        self.X_train, self.y_train = self._load_data(
            self.config.train_X_path, self.config.train_y_path
        )

        # Define hyperparameter grid
        self.param_grid: dict = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        }

    def _load_data(self, X_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load training data from the provided file paths."""
        try:
            logger.info(f"Loading data from {X_path} and {y_path}...")
            X: np.ndarray = np.load(X_path)
            y: np.ndarray = np.load(y_path)
            return X, y
        except FileNotFoundError as e:
            logger.error(f"File not found: {X_path} or {y_path}")
            raise e
        except Exception as e:
            logger.exception("Error loading data")
            raise e

    def _run_grid_search(self) -> GridSearchCV:
        """Perform grid search to find the best hyperparameters for XGBoost."""
        model = XGBClassifier()
        grid_search = GridSearchCV(
            model,
            param_grid=self.param_grid,
            cv=StratifiedKFold(n_splits=5),
            n_jobs=-1,
            verbose=2,
        )

        grid_search.fit(self.X_train, self.y_train)
        return grid_search

    def _save_best_model(self, best_model: XGBClassifier, best_params: dict) -> None:
        """Save the best model and hyperparameters to disk."""
        try:
            model_path = os.path.join(self.config.root_dir, "model.pkl")
            joblib.dump(best_model, model_path)
            logger.info(f"Best model saved at: {model_path}")

            params_path = os.path.join(
                self.config.root_dir, "best_hyperparameters.json"
            )
            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=4)
            logger.info(f"Best hyperparameters saved at: {params_path}")
        except Exception as e:
            logger.exception("Error saving model or hyperparameters")
            raise e

    def run(self) -> None:
        """Train the model, find the best one, and save it."""
        grid_search = self._run_grid_search()
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        if best_model:
            self._save_best_model(best_model, best_params)
            logger.info(f"Best model found with score: {grid_search.best_score_}")
        else:
            logger.error("No best model found during grid search.")


if __name__ == "__main__":
    try:
        config = ConfigurationManager().get_model_training_config()
        trainer = ModelTraining(config=config)
        trainer.run()
    except Exception as e:
        logger.exception("Model training pipeline failed")
        raise
