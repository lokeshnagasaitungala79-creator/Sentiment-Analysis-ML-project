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

            # Load features and labels together
            X_train, y_train = load_object(
                os.path.join("artifacts", "train_features.pkl")
            )

            X_test, y_test = load_object(
                os.path.join("artifacts", "test_features.pkl")
            )

            # Train the model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Predictions
            prediction = model.predict(X_test)

            # Evaluation
            accuracy = accuracy_score(y_test, prediction)
            f1 = f1_score(
                y_test, prediction, average="weighted", zero_division=0
            )

            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")

            print("\nClassification Report:")
            print(
                classification_report(
                    y_test,
                    prediction,
                    labels=[0, 1],
                    target_names=["Negative", "Positive"],
                    zero_division=0
                )
            )

            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, prediction))

            # Save trained model
            save_object(
                os.path.join("artifacts", "model.pkl"),
                model
            )

            logger.info("Model Training Completed")
            print("\nModel saved successfully!")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_Model_trainer()
