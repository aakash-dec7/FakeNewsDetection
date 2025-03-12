from src.fnd.constant import *
from src.fnd.utils.utils import read_yaml, create_directories
from src.fnd.entity.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataPreprocessingConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    PredictionConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_path=CONFIG_FILE_PATH,
        params_path=PARAMS_FILE_PATH,
        schema_path=SCHEMA_FILE_PATH,
    ):
        """
        Initializes the ConfigManager by reading configuration, parameters, and schema files.
        Creates necessary directories for storing artifacts.
        """

        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.schema = read_yaml(schema_path)

        create_directories(self.config.artifacts_root)

    ### Data Ingestion
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        ingestion_config = self.config.data_ingestion

        create_directories(ingestion_config.root_dir)

        return DataIngestionConfig(
            root_dir=ingestion_config.root_dir,
            download_path=ingestion_config.download_path,
        )

    ### Data Validation
    def get_data_validation_config(self) -> DataValidationConfig:
        validation_config = self.config.data_validation
        validation_schema = self.schema.columns

        return DataValidationConfig(
            train_data_path=validation_config.train_data_path,
            test_data_path=validation_config.test_data_path,
            valid_data_path=validation_config.valid_data_path,
            schema=validation_schema,
        )

    ### Data Preprocessing
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        preprocessing_config = self.config.data_preprocessing

        create_directories(preprocessing_config.root_dir)

        return DataPreprocessingConfig(
            root_dir=preprocessing_config.root_dir,
            train_data_path=preprocessing_config.train_data_path,
            test_data_path=preprocessing_config.test_data_path,
            vectorizer_path=preprocessing_config.vectorizer_path,
        )

    ### Data Transformation
    def get_data_transformation_config(self) -> DataTransformationConfig:
        transformation_config = self.config.data_transformation

        create_directories(transformation_config.root_dir)

        return DataTransformationConfig(
            root_dir=transformation_config.root_dir,
            train_data_path=transformation_config.train_data_path,
            test_data_path=transformation_config.test_data_path,
        )

    ### Model Training
    def get_model_training_config(self) -> ModelTrainingConfig:
        training_config = self.config.model_training

        create_directories(training_config.root_dir)

        return ModelTrainingConfig(
            root_dir=training_config.root_dir,
            train_X_path=training_config.train_X_path,
            train_y_path=training_config.train_y_path,
        )

    ### Model Evaluation
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        evaluation_config = self.config.model_evaluation
        exp_tracking_config = self.config.experiment_tracking

        create_directories(evaluation_config.root_dir)

        return ModelEvaluationConfig(
            root_dir=evaluation_config.root_dir,
            test_X_path=evaluation_config.test_X_path,
            test_y_path=evaluation_config.test_y_path,
            model_path=evaluation_config.model_path,
            metrics_path=evaluation_config.metrics_path,
            repo_name=exp_tracking_config.repo_name,
            repo_owner=exp_tracking_config.repo_owner,
            mlflow_uri=exp_tracking_config.mlflow.uri,
        )

    ### Prediction
    def get_prediction_config(self) -> PredictionConfig:
        prediction_config = self.config.prediction

        return PredictionConfig(
            model_path=prediction_config.model_path,
            vectorizer_path=prediction_config.vectorizer_path,
        )
