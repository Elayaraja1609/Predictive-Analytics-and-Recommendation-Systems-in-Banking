# Deployment Guide

## Prerequisites

- Python 3.10+
- pip

## 1. Setup

```bash
cd "Final Project"
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 2. Train all models

```bash
python run_pipeline.py
```

This will:

1. Generate synthetic datasets (Faker) under `data/raw/`
2. Load data into SQLite (`data/banking.db`)
3. Train loan default, segmentation, and recommendation models
4. Save metrics and plots under `reports/` and models under `models/`

## 3. Local deployment (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

## 4. Cloud deployment (Streamlit Community Cloud)

1. Push the project to GitHub (exclude `venv/`, `__pycache__/`, large binaries).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repo and set:
   - **Main file path:** `app/streamlit_app.py`
   - **Python version:** 3.10+
4. Add a `packages.txt` or use `requirements.txt` for dependencies.
5. Run `python run_pipeline.py` locally first and commit `models/` + `data/raw/` OR add a startup script that runs the pipeline on first deploy.

> **Note:** For cloud demo, pre-train models locally and commit `models/` and `reports/` folders, or add pipeline execution to Streamlit startup with caching.

## 5. SQL analytics

```bash
python -m src.sql_setup
```

Queries are defined in `sql/banking_analytics.sql`.

## 6. Submission zip contents

- `notebooks/` — EDA
- `src/` — source code
- `sql/` — analytics queries
- `app/` — Streamlit deployment
- `data/raw/` — datasets
- `models/` — trained artifacts
- `reports/` — metrics & visualizations
- `docs/` — deployment & documentation
- `README.md`
