import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, load_object


class ModelTrainer:
    def initiate_Model_trainer(self):
        try:
            logger.info("Model Training Started....")

            # Load transformed training data
            X_train = load_object("artifacts/train_features.pkl")
            y_train = load_object("artifacts/train_labels.pkl")

            # Load transformed test data
            X_test = load_object("artifacts/test_features.pkl")
            y_test = load_object("artifacts/test_labels.pkl")

            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Make predictions
            prediction = model.predict(X_test)

            # Evaluate model
            accuracy = accuracy_score(y_test, prediction)
            f1 = f1_score(
                y_test, prediction, average="weighted", zero_division=0
            )
            confusion_mat = confusion_matrix(y_test, prediction)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")

            print("\nClassification Report:")
            print(
                classification_report(
                    y_test,
                    prediction,
                    zero_division=0
                )
            )

            print("\nConfusion Matrix:")
            print(confusion_mat)

            # Save trained model
            os.makedirs("artifacts", exist_ok=True)
            save_object("artifacts/model.pkl", model)

            logger.info("Model Training Completed")
            print("\nModel saved successfully!")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_Model_trainer()
