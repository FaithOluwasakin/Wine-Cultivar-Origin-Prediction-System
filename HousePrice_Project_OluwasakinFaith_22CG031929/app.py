from flask import Flask, render_template, request
import os
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "house_price_model.pkl")
if os.path.exists(MODEL_PATH):
    model_obj = joblib.load(MODEL_PATH)
    pipeline = model_obj.get("pipeline")
    features = model_obj.get("features")
else:
    pipeline = None
    features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "FullBath", "YearBuilt"]


@app.route("/", methods=["GET", "POST"])
def index():
    global pipeline
    result = None
    if request.method == "POST":
        # Try loading model if not loaded
        if pipeline is None and os.path.exists(MODEL_PATH):
            try:
                model_obj = joblib.load(MODEL_PATH)
                pipeline = model_obj.get("pipeline")
            except Exception:
                pipeline = None

        # Read inputs
        vals = {}
        for f in features:
            val = request.form.get(f)
            vals[f] = float(val) if val not in (None, "") else 0.0

        df = pd.DataFrame([vals])

        if pipeline is None:
            result = "Model not found. Run model_development.py to train and save the model."
        else:
            pred = pipeline.predict(df)[0]
            result = f"Predicted SalePrice: {pred:,.2f}"

    return render_template("index.html", result=result, features=features)


if __name__ == "__main__":
    app.run(debug=True)
    # Run on a different port to avoid colliding with other project servers (e.g., Titanic app)
    app.run(debug=True, port=5001)
