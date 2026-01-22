from flask import Flask, render_template, request
import os
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "wine_cultivar_model.pkl")
if os.path.exists(MODEL_PATH):
    model_obj = joblib.load(MODEL_PATH)
    pipeline = model_obj.get("pipeline")
    features = model_obj.get("features")
    target_names = model_obj.get("target_names")
else:
    pipeline = None
    features = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols"]
    target_names = None


@app.route("/", methods=["GET", "POST"])
def index():
    global pipeline
    result = None
    if request.method == "POST":
        # Attempt to load model if missing
        if pipeline is None and os.path.exists(MODEL_PATH):
            try:
                model_obj = joblib.load(MODEL_PATH)
                pipeline = model_obj.get("pipeline")
            except Exception:
                pipeline = None

        vals = {}
        for f in features:
            v = request.form.get(f)
            vals[f] = float(v) if v not in (None, "") else 0.0

        df = pd.DataFrame([vals])

        if pipeline is None:
            result = "Model not found. Run model_building.py to train the model."
        else:
            pred = pipeline.predict(df)[0]
            label = target_names[pred] if target_names is not None else str(pred)
            result = f"Predicted cultivar: {label} (class {pred})"

    return render_template("index.html", result=result, features=features)


if __name__ == "__main__":
    app.run(debug=True, port=5003)
