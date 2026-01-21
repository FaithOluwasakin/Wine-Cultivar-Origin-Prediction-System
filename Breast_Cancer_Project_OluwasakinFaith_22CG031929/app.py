import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "model.joblib")
model_obj = joblib.load(MODEL_PATH)
pipeline = model_obj["pipeline"]
features = model_obj["features"]
target_names = model_obj.get("target_names", ["malignant", "benign"]) 


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", features=features)


@app.route("/predict", methods=["POST"])
def predict():
    values = []
    for f in features:
        v = request.form.get(f, "0")
        try:
            values.append(float(v))
        except ValueError:
            values.append(0.0)

    arr = np.array(values).reshape(1, -1)
    pred = pipeline.predict(arr)[0]
    label = target_names[pred].capitalize()

    return render_template("index.html", features=features, result=label, inputs=dict(zip(features, values)))


if __name__ == "__main__":
    app.run(debug=True)
