# 🚗 US Accidents EDA Portfolio

A comprehensive Machine Learning & Exploratory Data Analysis portfolio built on the [US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) — 7.7 million accident records across 49 US states (Feb 2016 – Mar 2023).

## 🏗️ Stack

| Layer | Technology |
|-------|-----------|
| Data | pandas, NumPy, PyArrow |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML | scikit-learn, XGBoost, LightGBM |
| Backend | FastAPI + SQLAlchemy + PostgreSQL |
| **Auth/DB** | **SQLite + Werkzeug (Hashing)** |
| Frontend | Dash + Bootstrap (DARKLY) |
| DevOps | Docker, GitHub Actions CI |

## 📁 Project Structure

```
ml-eda-portfolio/
├── configs/            # config.yaml, model_config.yaml
├── data/raw/           # US_Accidents_March23.csv 
├── src/
│   ├── data/           # make_dataset, clean_data, build_features
│   ├── models/         # train_model, evaluate_model, predict_model
│   ├── visualization/  # visualize.py (static + interactive plots)
│   └── utils/          # logger, helpers
├── app/
│   ├── backend/        # FastAPI, auth_db.py, users.db (SQLite)
│   └── frontend/       # Dash dashboard (dashboard.py), assets/
├── notebooks/          # 6 numbered EDA notebooks
├── tests/              # pytest test suite
└── docs/               # Architecture, API docs, deployment guide
```

## 🚀 Quick Start

```powershell
# 1. Activate environment
& .mlVenv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Run full pipeline (takes ~30-60 min on full dataset)
python src/data/make_dataset.py
python src/data/clean_data.py
python src/data/build_features.py
python src/models/train_model.py
python src/models/evaluate_model.py

# 4. Launch app
python app/frontend/dashboard.py
```

- **Dashboard**: http://localhost:8050
- **API Docs**: http://localhost:8000/docs

> **Dev tip**: Set `sample_size: 500000` in `configs/config.yaml` for faster iteration.

## 🔐 Authentication & Access

The application now includes a secure entry flow:
1. **Landing Page**: Overview of project capabilities and stats.
2. **Sliding Auth**: A modern, animated interface for Login and Signup.
3. **Session Management**: Access to the interactive dashboard is restricted to authenticated users.

## 📊 Dashboard Pages

1. **Landing** – Project hero, features, and national impact overview
2. **Auth** – Secure sliding login/signup interface
3. **Overview** – KPI cards, severity distribution, accidents over time, state choropleth map
4. **EDA** – Interactive analysis: hourly patterns, weather impact, top cities, road features
5. **ML Models** – Metrics table, bar chart comparison, radar chart
6. **Predict** – Real-time severity prediction with probability gauge

## 🤖 ML Models

| Model | Task | Handles Imbalance |
|-------|------|-------------------|
| Logistic Regression | Binary classification (High/Low severity) | `class_weight=balanced` |
| Random Forest | Binary classification | `class_weight=balanced` |
| XGBoost | Binary classification | `scale_pos_weight` |
| LightGBM | Binary classification | `class_weight=balanced` |

**Target**: Severity ≥ 3 → High (1), else Low (0)

## 🧪 Tests

```powershell
pytest tests/ -v
```

## 📖 Documentation

- [Architecture](docs/architecture.md)
- [API Documentation](docs/api_documentation.md)
- [Deployment Guide](docs/deployment_guide.md)

## 📄 License

MIT License — see [LICENSE](LICENSE)
