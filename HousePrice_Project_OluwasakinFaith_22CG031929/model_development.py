import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_dataset(local_names=None):
    local_names = local_names or ["house_prices.csv", "HousePrice.csv", "train.csv"]
    for name in local_names:
        if os.path.exists(name):
            print("Loading local dataset:", name)
            return pd.read_csv(name)

    # fallback: attempt to download a public Ames Housing mirror
    urls = [
        "https://raw.githubusercontent.com/selva86/datasets/master/AmesHousing.csv",
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/ames/ames.csv",
    ]
    for url in urls:
        try:
            print("Attempting download from", url)
            df = pd.read_csv(url)
            print("Downloaded dataset from", url)
            return df
        except Exception as e:
            print("Download failed for", url, "->", str(e))

    raise FileNotFoundError(
        "No dataset found locally and downloads failed. Please download the Ames dataset (train.csv) and place it as 'house_prices.csv' in the project root."
    )


def build_and_save(output_path="model/house_price_model.pkl"):
    df = load_dataset()

    # Choose six features from the allowed nine
    features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "FullBath", "YearBuilt"]
    target = "SalePrice"

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError("Dataset missing required columns: " + ",".join(missing))

    df = df[features + [target]].copy()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, features)])

    model = Pipeline(steps=[("pre", preprocessor), ("reg", RandomForestRegressor(n_estimators=100, random_state=42))])

    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({"pipeline": model, "features": features}, output_path)
    print("Saved model to", output_path)


if __name__ == "__main__":
    build_and_save()

