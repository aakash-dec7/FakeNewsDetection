import numpy as np
from main import model, vectorizer
from src.fnd.logger import logger
from src.fnd.entity.entity import PredictionConfig


class Prediction:
    def __init__(self, config: PredictionConfig):
        """Initializes the Prediction class with configuration settings."""
        self.config = config
        self.model = model
        self.vectorizer = vectorizer

    def _preprocess_input(self, text: str) -> np.ndarray:
        """Transforms input text into a numerical representation for the model."""
        try:
            text = text.strip()
            if not text:
                logger.warning("Received empty text input.")
                return np.zeros((1, self.vectorizer.vector_size))

            tokens = text.split()  # Use a better tokenizer if needed
            vectors = [
                self.vectorizer.wv[word]
                for word in tokens
                if word in self.vectorizer.wv
            ]

            return (
                np.mean(vectors, axis=0).reshape(1, -1)
                if vectors
                else np.zeros((1, self.vectorizer.vector_size))
            )
        except Exception as e:
            logger.exception(f"Error in preprocessing input: {e}")
            return np.zeros((1, self.vectorizer.vector_size))

    def _map_prediction_to_label(self, prediction: np.ndarray) -> str:
        """Maps the model's output to a human-readable label."""
        liar_labels = {
            0: "True",
            1: "Mostly-True",
            2: "Half-True",
            3: "Barely-True",
            4: "False",
            5: "Pants-on-Fire",
        }
        predicted_class = (
            int(prediction.argmax()) if prediction.ndim > 1 else int(prediction[0])
        )
        return liar_labels.get(predicted_class, "UNKNOWN")

    def predict(self, text: str) -> str:
        """Generates a fake news prediction label for the given text."""
        try:
            logger.info("Starting prediction process...")
            processed_text = self._preprocess_input(text)
            prediction = self.model.predict(processed_text)
            logger.debug(f"Raw model output: {prediction}")
            result = self._map_prediction_to_label(prediction)
            logger.info(f"Prediction result: {result}")
            return result
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return ""
