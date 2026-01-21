import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def build_and_save_model(output_path="models/model.joblib"):
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # Normalize column names: convert 'mean radius' -> 'radius_mean'
    def normalize(name):
        if name.startswith("mean "):
            return name.replace("mean ", "") + "_mean"
        return name.replace(" ", "_")

    X.columns = [normalize(c) for c in X.columns]

    y = pd.Series(data.target)  # 0=malignant, 1=benign

    # Select five input features as required
    features = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
    ]

    X = X[features]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline: scaler + logistic regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Evaluation on test set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Full classification report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

    # Save pipeline and metadata
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    to_save = {
        "pipeline": pipe,
        "features": features,
        "target_names": list(data.target_names),
    }
    joblib.dump(to_save, output_path)
    print(f"Saved model pipeline and metadata to {output_path}")


def load_model_and_predict(sample, model_path="models/model.joblib"):
    model_obj = joblib.load(model_path)
    pipe = model_obj["pipeline"]
    features = model_obj["features"]
    target_names = model_obj.get("target_names", ["malignant", "benign"])

    arr = np.array(sample).reshape(1, -1)
    pred = pipe.predict(arr)[0]
    return target_names[pred]


if __name__ == "__main__":
    build_and_save_model()

    # Demonstrate reload and single prediction
    model_path = "models/model.joblib"
    model_obj = joblib.load(model_path)
    # Demo prediction on a neutral sample (zeros after scaling)
    print("Demo prediction on zero-centered sample:")
    print(load_model_and_predict([0.0] * len(model_obj["features"])) )
