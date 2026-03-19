# Deployment Guide

## Local Development

### 1. Setup environment
```powershell
# Activate venv
& d:\ml-eda-portfolio\.mlVenv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment
```powershell
copy .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 3. Run the full pipeline
```powershell
# Step 1: Load raw data → interim Parquet
python src/data/make_dataset.py

# Step 2: Clean data
python src/data/clean_data.py

# Step 3: Feature engineering
python src/data/build_features.py

# Step 4: Train models
python src/models/train_model.py

# Step 5: Evaluate models
python src/models/evaluate_model.py
```

### 4. Start the application
```powershell
# Option A: Start both services together
python app/app.py

# Option B: Start separately
# Terminal 1 - Backend
python -m uvicorn app.backend.main:app --reload --port 8000

# Terminal 2 - Frontend
python app/frontend/dashboard.py
```

Access:
- Dashboard: http://localhost:8050
- API docs: http://localhost:8000/docs

### 5. Run tests
```powershell
pytest tests/ -v
```

## Docker Deployment

```bash
# Build image
docker build -f app/Dockerfile -t ml-eda-portfolio .

# Run with PostgreSQL
docker run -p 8000:8000 -p 8050:8050 \
  -e DB_HOST=host.docker.internal \
  -e DB_PASSWORD=your_password \
  ml-eda-portfolio
```

## PostgreSQL Setup

```sql
CREATE DATABASE accidents_db;
CREATE USER portfolio_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE accidents_db TO portfolio_user;
```

## Performance Notes

- The raw CSV is ~2.9 GB (7.7M rows). `make_dataset.py` reads it in 100K-row chunks.
- For development, set `sample_size: 500000` in `configs/config.yaml`.
- Training XGBoost/LightGBM on the full dataset takes ~10-30 min depending on hardware.
- Processed Parquet is ~200-400 MB (much faster to load than CSV).
