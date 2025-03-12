import os
import re
import nltk
import json
import string
import pandas as pd
import numpy as np
from pathlib import Path
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from src.fnd.logger import logger
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from src.fnd.entity.entity import DataPreprocessingConfig
from src.fnd.config.configuration import ConfigurationManager


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        """Initialize data preprocessing configuration."""
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.label_encoder = LabelEncoder()

        # Ensure NLTK resources are available
        nltk_dir = os.path.join(self.config.root_dir, "nltk_data")
        os.makedirs(nltk_dir, exist_ok=True)
        nltk.data.path.append(nltk_dir)

        # Download required NLTK resources
        for resource in ["punkt", "punkt_tab", "wordnet", "stopwords"]:
            nltk.download(resource, download_dir=Path(nltk_dir))

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load dataset from CSV."""
        try:
            logger.info(f"Loading dataset from: {file_path}")
            return pd.read_csv(file_path)[["statement", "label"]].dropna()
        except Exception as e:
            logger.exception(f"Error loading dataset: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> list:
        """Tokenize, clean, and lemmatize text."""
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = "".join([char for char in text if not char.isdigit()])
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub("\\s+", " ", text).strip()
        tokens = word_tokenize(text)
        processed_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and word.isalnum()
        ]
        return processed_tokens

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the text data."""
        logger.info("Cleaning data: Removing missing values and processing text.")
        data["statement"] = data["statement"].astype(str).apply(self._preprocess_text)
        return data

    def _vectorize_data(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> tuple:
        """Convert text data into Word2Vec embeddings."""
        logger.info("Training Word2Vec model and transforming datasets.")

        sentences = train_data["statement"].tolist()
        vector_size = 100
        vectorizer = Word2Vec(
            sentences, vector_size=vector_size, window=5, min_count=1, workers=4
        )

        def get_avg_word_vector(tokens, model):
            vectors = [model.wv[word] for word in tokens if word in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

        train_data["statement"] = train_data["statement"].apply(
            lambda x: get_avg_word_vector(x, vectorizer)
        )
        test_data["statement"] = test_data["statement"].apply(
            lambda x: get_avg_word_vector(x, vectorizer)
        )

        train_data["label"] = self.label_encoder.fit_transform(train_data["label"])
        test_data["label"] = self.label_encoder.transform(test_data["label"])

        os.makedirs(self.config.vectorizer_path, exist_ok=True)
        vectorizer_path = os.path.join(self.config.vectorizer_path, "vectorizer.bin")
        vectorizer.save(vectorizer_path)
        logger.info(f"Word2Vec model saved at {vectorizer_path}")

        return train_data, test_data

    def _save_preprocessed_data(self, data: pd.DataFrame, file_path: str):
        """Save preprocessed data with properly formatted JSON lists."""
        logger.info(f"Saving preprocessed data to {file_path}.")
        try:
            os.makedirs(self.config.root_dir, exist_ok=True)
            data["statement"] = data["statement"].apply(
                lambda x: json.dumps(x.tolist())
            )
            data.to_csv(file_path, index=False)
        except Exception as e:
            logger.exception(f"Error saving preprocessed data: {str(e)}")
            raise

    def run(self):
        """Execute the full data preprocessing pipeline."""
        try:
            train_data = self._load_data(self.config.train_data_path)
            test_data = self._load_data(self.config.test_data_path)

            train_data = self._clean_data(train_data)
            test_data = self._clean_data(test_data)

            train_vectorized, test_vectorized = self._vectorize_data(
                train_data, test_data
            )

            self._save_preprocessed_data(
                train_vectorized,
                os.path.join(self.config.root_dir, "train_preprocessed.csv"),
            )
            self._save_preprocessed_data(
                test_vectorized,
                os.path.join(self.config.root_dir, "test_preprocessed.csv"),
            )

            logger.info("Data preprocessing pipeline completed successfully.")
        except Exception as e:
            logger.exception("Data preprocessing pipeline failed.")
            raise RuntimeError("Data preprocessing pipeline failed.") from e


if __name__ == "__main__":
    try:
        data_preprocessing_config = (
            ConfigurationManager().get_data_preprocessing_config()
        )
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.run()
    except Exception as e:
        raise RuntimeError("Data preprocessing pipeline failed.") from e
