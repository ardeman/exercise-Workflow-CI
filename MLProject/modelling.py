from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_churn_preprocessing" / "customer_churn_preprocessed.csv"
TARGET_COLUMN = "churn"


def main() -> None:
    mlflow.set_experiment("Customer Churn CI Retraining")
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    with mlflow.start_run(run_name="ci-random-forest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, predictions))
        mlflow.log_metric("test_precision", precision_score(y_test, predictions, zero_division=0))
        mlflow.log_metric("test_recall", recall_score(y_test, predictions, zero_division=0))
        mlflow.log_metric("test_f1", f1_score(y_test, predictions, zero_division=0))
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=x_test.head(2))


if __name__ == "__main__":
    main()
