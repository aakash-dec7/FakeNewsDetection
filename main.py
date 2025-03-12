import joblib
import xgboost as xgb
from src.fnd.logger import logger
from src.fnd.config.configuration import ConfigurationManager


# Load prediction configuration
prediction_config = ConfigurationManager().get_prediction_config()

# Load XGBoost Model ONCE
try:
    logger.info(f"Loading XGBoost model from: {prediction_config.model_path}")
    model = joblib.load(prediction_config.model_path)
    logger.info("XGBoost model loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load XGBoost model: {e}")
    model = None

# Load Vectorizer ONCE
try:
    logger.info(f"Loading vectorizer from: {prediction_config.vectorizer_path}")
    vectorizer = joblib.load(prediction_config.vectorizer_path)
    logger.info("Vectorizer loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load vectorizer: {e}")
    vectorizer = None
