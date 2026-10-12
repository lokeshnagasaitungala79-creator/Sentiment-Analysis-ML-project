import os
import sys

from src.exception import CustomException
from src.logger import logger
from src.utils import clean_text, load_object


class PredictionPipeline:
    def __init__(self):
        try:
            self.model = load_object(
                os.path.join("artifacts", "model.pkl")
            )
            self.vectorizer = load_object(
                os.path.join("artifacts", "vectorizer.pkl")
            )
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, review):
        try:
            if not isinstance(review, str) or not review.strip():
                raise ValueError("Please enter a valid movie review.")

            cleaned_review = clean_text(review)

            transformed_review = self.vectorizer.transform(
                [cleaned_review]
            )

            prediction = self.model.predict(transformed_review)[0]

            result = "Positive" if prediction == 1 else "Negative"

            logger.info("Prediction pipeline completed successfully.")

            return result

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = PredictionPipeline()
    print(pipeline.predict("The movie was very bad"))
