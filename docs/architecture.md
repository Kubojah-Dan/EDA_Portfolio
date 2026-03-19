# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                            │
│  raw CSV → make_dataset → clean_data → build_features → parquet │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ML Pipeline                              │
│  processed parquet → train_model → evaluate_model → .pkl files  │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
         ┌──────────────────┐  ┌──────────────────┐
         │  FastAPI Backend │  │  PostgreSQL DB   │
         │  (port 8000)     │◄─►  (accidents_db)  │
         └──────────────────┘  └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │   Dash Frontend  │
         │   (port 8050)    │
         └──────────────────┘
```

## Components

### Data Pipeline (`src/data/`)
- `make_dataset.py` – Chunked CSV loading → typed interim Parquet
- `clean_data.py` – Dedup, missing value imputation, outlier capping, coordinate validation
- `build_features.py` – Temporal, weather, road features; label encoding; binary target

### ML Pipeline (`src/models/`)
- `train_model.py` – Trains LR, RF, XGBoost, LightGBM with class-imbalance handling
- `evaluate_model.py` – ROC/PR curves, confusion matrices, feature importance, calibration
- `predict_model.py` – Inference pipeline with model caching

### Backend (`app/backend/`)
- FastAPI with async endpoints
- SQLAlchemy ORM + PostgreSQL
- Pydantic v2 request/response validation
- Auto-generated Swagger docs at `/docs`

### Frontend (`app/frontend/`)
- Dash + Bootstrap (DARKLY theme)
- 4 pages: Overview, EDA, ML Models, Prediction Tool
- Communicates with backend via REST API

## Data Flow
1. Raw CSV (7.7M rows, ~2.9 GB) → chunked load → interim Parquet
2. Cleaning → same interim Parquet (overwrite)
3. Feature engineering → processed Parquet + feature_metadata.json
4. Training → model_*.pkl + training_summary.json
5. Evaluation → test_metrics.json + figures/
6. API serves processed Parquet + model files
7. Dashboard fetches from API endpoints
