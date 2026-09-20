"""
Bias Analyzer API
------------------
Loads the exact trained RandomForestClassifier + ColumnTransformer
preprocessor you uploaded, and serves predictions over a small REST API
for the accompanying HTML/CSS/JS frontend.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open index.html in your browser (or serve it with any static file
server) - it talks to this API at http://localhost:5000 by default.
"""

import logging
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bias-analyzer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_file(filename):
    """Look in model/<filename> first, then fall back to right next to app.py,
    so it works whether or not you kept the model/ subfolder."""
    candidates = [
        os.path.join(BASE_DIR, "model", filename),
        os.path.join(BASE_DIR, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename}. Looked in: {candidates}. "
        f"Put it in a 'model' subfolder next to app.py, or directly next to app.py."
    )


MODEL_PATH = find_file("bias_analyzer_model.pkl")
PREPROCESSOR_PATH = find_file("bias_analyzer_preprocessor.pkl")

app = Flask(__name__)

# Reject any request body over 32KB outright. A real payload here (14 short
# fields) is well under 1KB, so this only ever blocks abuse, never a real user.
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024

# CORS: defaults to "*" (any origin) for zero-friction local development.
# Before deploying this publicly, set ALLOWED_ORIGIN to your actual frontend's
# URL (e.g. https://your-username.github.io) so random sites can't call your
# model from a visitor's browser. This API has no authentication, so treat
# the origin restriction as your only access control once it's on the internet.
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
CORS(app, origins=ALLOWED_ORIGIN)

# Rate limiting: bias-check reruns the forest ~20x per call (once per race x
# sex combination), so it gets a tighter limit than the single-shot endpoints.
# Override via env vars if you need looser/tighter limits for your deployment.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[os.environ.get("RATE_LIMIT_DEFAULT", "60 per minute")],
    storage_uri="memory://",
)

log.info("Loading model + preprocessor...")
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
log.info("Model loaded.")

CATEGORICAL_COLS = ["workclass", "education", "marital_status", "occupation",
                     "relationship", "race", "sex", "native_country"]
NUMERIC_COLS = ["age", "fnlwgt", "education_num", "capital_gain",
                 "capital_loss", "hours_per_week"]
ALL_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# Pull the exact category options the model was trained on, straight from the
# fitted encoder, so the frontend dropdowns can never send an out-of-vocabulary value.
CAT_OPTIONS = {}
ohe = preprocessor.transformers_[0][1]
for col, cats in zip(preprocessor.transformers_[0][2], ohe.categories_):
    CAT_OPTIONS[col] = list(cats)


def to_dataframe(payload):
    row = {col: payload.get(col) for col in ALL_COLS}
    missing = [c for c in ALL_COLS if row[c] is None or row[c] == ""]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")
    for col in NUMERIC_COLS:
        try:
            row[col] = float(row[col])
        except (TypeError, ValueError):
            raise ValueError(f"Field '{col}' must be a number.")
    return pd.DataFrame([row])


def run_prediction(payload):
    df = to_dataframe(payload)
    X = preprocessor.transform(df)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    return {
        "prediction": pred,
        "label": ">50K" if pred == 1 else "<=50K",
        "probability_le_50k": proba[0],
        "probability_gt_50k": proba[1],
    }


@app.route("/api/options", methods=["GET"])
def options():
    """Tells the frontend exactly which categorical values are valid."""
    return jsonify({
        "categorical_options": CAT_OPTIONS,
        "numeric_fields": NUMERIC_COLS,
        "categorical_fields": CATEGORICAL_COLS,
    })


@app.route("/api/predict", methods=["POST"])
@limiter.limit(os.environ.get("RATE_LIMIT_PREDICT", "30 per minute"))
def predict():
    try:
        payload = request.get_json(force=True)
        result = run_prediction(payload)
        return jsonify(result)
    except ValueError as e:
        # Validation errors (e.g. missing field) are safe to show as-is —
        # they only ever describe the user's own input, never internals.
        return jsonify({"error": str(e)}), 400
    except HTTPException:
        raise  # let Flask's own handlers deal with 413, 429, etc.
    except Exception:
        log.exception("Unexpected error in /api/predict")
        return jsonify({"error": "Could not process this request. Check your input and try again."}), 400


@app.route("/api/bias-check", methods=["POST"])
@limiter.limit(os.environ.get("RATE_LIMIT_BIAS_CHECK", "10 per minute"))
def bias_check():
    """
    Holds every field fixed EXCEPT race and sex, and re-runs the model across
    every combination of race x sex the model was trained on. This isolates
    the effect of those two protected attributes on the model's output,
    which is the core "bias analysis" use case.
    """
    try:
        payload = request.get_json(force=True)
        base = to_dataframe(payload).iloc[0].to_dict()

        rows = []
        combos = []
        for race in CAT_OPTIONS["race"]:
            for sex in CAT_OPTIONS["sex"]:
                r = dict(base)
                r["race"] = race
                r["sex"] = sex
                rows.append(r)
                combos.append((race, sex))

        df = pd.DataFrame(rows)
        X = preprocessor.transform(df)
        preds = model.predict(X)
        probas = model.predict_proba(X)

        results = []
        for (race, sex), pred, proba in zip(combos, preds, probas):
            results.append({
                "race": race,
                "sex": sex,
                "prediction": int(pred),
                "label": ">50K" if pred == 1 else "<=50K",
                "probability_gt_50k": float(proba[1]),
            })

        gt50 = [r["probability_gt_50k"] for r in results]
        spread = max(gt50) - min(gt50)

        return jsonify({"results": results, "max_probability_spread": spread})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except HTTPException:
        raise  # let Flask's own handlers deal with 413, 429, etc.
    except Exception:
        log.exception("Unexpected error in /api/bias-check")
        return jsonify({"error": "Could not process this request. Check your input and try again."}), 400


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.errorhandler(413)
def payload_too_large(e):
    return jsonify({"error": "Request body too large."}), 413


@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"error": "Too many requests. Please slow down and try again shortly."}), 429


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
