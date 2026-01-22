import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def build_and_save(output_path="model/wine_cultivar_model.pkl"):
    data = load_wine()
    X_all = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Select six features from allowed list
    features = [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
    ]

    X = X_all[features].copy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline: scaling + classifier
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds, target_names=data.target_names))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": features, "target_names": list(data.target_names)}, output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    build_and_save()
