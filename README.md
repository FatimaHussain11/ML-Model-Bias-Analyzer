# ⚖️ Income Bias Analyzer

**A fairness audit tool for a Census-income classifier.** Fix an applicant's profile, then watch the model's predicted income band shift as race and sex change — everything else held constant.

<img width="1152" height="844" alt="Capture" src="https://github.com/user-attachments/assets/c822b302-33d0-454b-a35e-2dbcee9cc895" />
<img width="681" height="816" alt="image" src="https://github.com/user-attachments/assets/234d00f1-322b-4a25-9906-7a4e50b57a2f" />
<img width="437" height="203" alt="image" src="https://github.com/user-attachments/assets/49ea313c-d355-45e4-8305-a079665f9984" />
<img width="655" height="521" alt="image" src="https://github.com/user-attachments/assets/111cd961-a561-47ff-ac15-799f4c2943bc" />

---

## What this is

A `RandomForestClassifier` (100 trees) trained on Census-style income data predicts whether someone earns above or below $50K/year, using features like age, education, occupation — and **race** and **sex**.

This app isolates the effect of those two protected attributes: it takes one applicant's profile, holds every other field fixed, and reruns the model across every race × sex combination it was trained on. If the predicted probability swings meaningfully just from changing race or sex, that's the kind of signal a fairness audit is built to catch.

## How it works

```
┌─────────────────┐      REST      ┌──────────────────┐      joblib      ┌─────────────────────┐
│  index.html      │ ─────────────▶ │   Flask API       │ ───────────────▶ │  RandomForestClassifier │
│  style.css        │  fetch()        │   (app.py)         │                   │  + ColumnTransformer    │
│  script.js         │ ◀───────────── │                    │ ◀─────────────── │  (your trained model)     │
└─────────────────┘   JSON          └──────────────────┘   predict_proba   └─────────────────────┘
```

The forest has ~920,000 decision nodes — too large to run client-side in a browser without a multi-megabyte payload — so a small Flask API loads the real `.pkl` files and serves predictions over two endpoints:

| Endpoint | What it does |
|---|---|
| `POST /api/predict` | Runs the model on one applicant profile, returns the predicted band and confidence |
| `POST /api/bias-check` | Holds the profile fixed, varies race × sex, returns every combination's probability and the max spread between them |
| `GET /api/options` | Returns the exact categorical values the model was trained on, so the frontend never sends an out-of-vocabulary value |

## Quick start

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# this repo uses Git LFS for the model file — install it once if you haven't:
git lfs install
git lfs pull

pip install -r requirements.txt
python app.py
```

Then open `index.html` in your browser. The page defaults to `http://localhost:5000` — edit the "API endpoint" field top-right if you run the API somewhere else.

> **Note:** the model was trained with `scikit-learn==1.6.1`. `requirements.txt` pins that exact version so the pickle files load cleanly.

## Deploying it publicly

The frontend and backend are decoupled, so you can host them separately:

- **Backend** (`app.py`) — deploy to [Render](https://render.com), [Railway](https://railway.app), or [Fly.io](https://fly.io). Build command: `pip install -r requirements.txt`. Start command: `python app.py`.
- **Frontend** (`index.html`, `style.css`, `script.js`) — host as static files on GitHub Pages, Netlify, or Vercel, then point the "API endpoint" field at your deployed backend's URL.

Before making it public, set the `ALLOWED_ORIGIN` environment variable on your backend to your frontend's exact URL (see [Security](#-security-notes) below) — otherwise any website can call your model from a visitor's browser.

## 🔒 Security notes

This project has no authentication by design (it's a local analysis tool), so if you deploy the API publicly, keep these in mind:

- **CORS** — `app.py` reads `ALLOWED_ORIGIN` from the environment (defaults to `*` for local dev). Set it to your actual frontend origin before going public:
  ```bash
  export ALLOWED_ORIGIN=https://your-username.github.io
  ```
- **Rate limiting** — built in via `flask-limiter`: 30 requests/minute on `/api/predict`, 10/minute on `/api/bias-check` (it reruns the model ~20x per call), 60/minute default elsewhere. Override with `RATE_LIMIT_PREDICT`, `RATE_LIMIT_BIAS_CHECK`, `RATE_LIMIT_DEFAULT` env vars. The in-memory limiter resets if the process restarts and doesn't share state across multiple server instances — swap `storage_uri` in `app.py` for Redis if you scale beyond one process.
- **Request size limit** — bodies over 32KB are rejected outright (a real request here is under 1KB), so oversized payloads can't be used to exhaust memory.
- **Sanitized error messages** — input-validation errors (e.g. "missing field") are shown to the caller since they only ever describe their own request; anything unexpected is logged server-side with a full traceback and returns a generic message to the client, so internal details never leak in a response.
- **Debug mode is off** (`debug=False`) — keep it that way in any public deployment; Flask's debugger allows arbitrary code execution if left on.
- **Model file via Git LFS** — the `.pkl` files are tracked with [Git LFS](https://git-lfs.com) (`.gitattributes`), not committed as raw blobs, since GitHub rejects files over 100MB and discourages large binaries in normal history.

## Project structure

```
.
├── app.py                              # Flask API — loads the model, serves predictions
├── requirements.txt                    # Pinned Python dependencies
├── index.html / style.css / script.js  # Frontend
├── screenshots/                        # README preview images
├── model/
│   ├── bias_analyzer_model.pkl         # RandomForestClassifier (tracked via Git LFS)
│   └── bias_analyzer_preprocessor.pkl  # ColumnTransformer (one-hot + passthrough)
├── .gitattributes                      # Git LFS config
├── .gitignore
└── LICENSE
```

## License

MIT — see [LICENSE](LICENSE). The model itself was trained on Census/Adult-style income data; check your own rights before redistributing the trained weights if you didn't train them yourself.
