
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


class DataIngestion:

    def initiate_data_ingestion(self):
        try:
            # Get the project root directory
            project_root = Path(__file__).resolve().parents[2]

            # Dataset location: Note Book/sentiment_data.csv
            data_path = (
                project_root
                / "Note Book"
                / "sentiment_data.csv"
            )

            # Create artifacts directory
            artifact_dir = project_root / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Check whether dataset exists
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found: {data_path}"
                )

            # Load dataset
            movie_reviews = pd.read_csv(data_path)

            logger.info("Dataset loaded successfully")

            # Validate required columns
            required_columns = {"review", "sentiment"}

            if not required_columns.issubset(movie_reviews.columns):
                raise ValueError(
                    "CSV must contain 'review' and 'sentiment' columns. "
                    f"Found columns: {list(movie_reviews.columns)}"
                )

            # Keep required columns and remove missing values
            movie_reviews = movie_reviews[
                ["review", "sentiment"]
            ].dropna().copy()

            # Clean basic column values
            movie_reviews["review"] = (
                movie_reviews["review"].astype(str).str.strip()
            )

            movie_reviews["sentiment"] = (
                movie_reviews["sentiment"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            # Keep only positive and negative reviews
            movie_reviews = movie_reviews[
                movie_reviews["sentiment"].isin(
                    ["positive", "negative"]
                )
            ].copy()

            # Remove empty reviews
            movie_reviews = movie_reviews[
                movie_reviews["review"] != ""
            ].copy()

            # Validate dataset
            if movie_reviews.empty:
                raise ValueError(
                    "No valid positive or negative reviews found."
                )

            if movie_reviews["sentiment"].nunique() != 2:
                raise ValueError(
                    "Dataset must contain both positive and negative reviews."
                )

            logger.info(
                "Valid dataset contains %s rows",
                len(movie_reviews)
            )

            # Split dataset into training and testing sets
            train_movie_reviews, test_movie_reviews = train_test_split(
                movie_reviews,
                test_size=0.2,
                random_state=42,
                stratify=movie_reviews["sentiment"]
            )

            # Save train and test datasets
            train_path = artifact_dir / "train.csv"
            test_path = artifact_dir / "test.csv"

            train_movie_reviews.to_csv(
                train_path, index=False
            )

            test_movie_reviews.to_csv(
                test_path, index=False
            )

            logger.info(
                "Data ingestion completed. Train: %s, Test: %s",
                len(train_movie_reviews),
                len(test_movie_reviews)
            )

            return str(train_path), str(test_path)

        except Exception as e:
            logger.exception("Data ingestion failed")
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    ingestion = DataIngestion()
    print(ingestion.initiate_data_ingestion())