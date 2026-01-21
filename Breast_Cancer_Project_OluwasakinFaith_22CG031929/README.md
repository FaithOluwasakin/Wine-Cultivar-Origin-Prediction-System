# Breast Cancer Prediction Project

This project trains a Logistic Regression classifier on the Breast Cancer Wisconsin (Diagnostic) dataset and exposes a small Flask web UI to make predictions using five selected features.

Selected features:
- `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`

Quick start:

1. Create a Python environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Train and save the model:

```bash
python model_development.py
```

This creates `models/model.joblib`.

3. Run the Flask app:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser to use the form.

Note: This is for educational purposes only and not a medical diagnostic tool.
