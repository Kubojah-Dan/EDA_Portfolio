# API Documentation

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

## Endpoints

### System
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check, lists loaded models |

### ML
| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Predict accident severity |
| GET | `/metrics` | Get test metrics for all models |

### Analytics
| Method | Path | Description |
|--------|------|-------------|
| GET | `/stats` | Dataset-level statistics |

### EDA
| Method | Path | Description |
|--------|------|-------------|
| GET | `/eda/severity-by-hour` | Accident counts by hour and severity |
| GET | `/eda/severity-by-state` | Avg severity per state |
| GET | `/eda/accidents-over-time` | Monthly accident counts |
| GET | `/eda/weather-impact` | Weather score vs severity |
| GET | `/eda/top-cities?top_n=15` | Top cities by accident count |
| GET | `/eda/road-features` | Road feature impact on accidents |

## POST /predict – Request Body

```json
{
  "hour": 8,
  "day_of_week": 1,
  "month": 3,
  "year": 2023,
  "is_weekend": 0,
  "is_rush_hour": 1,
  "is_night": 0,
  "season": 1,
  "duration_min": 30.0,
  "weather_severity_score": 1.0,
  "road_feature_count": 2,
  "Distance(mi)": 0.5,
  "Temperature(F)": 55.0,
  "Humidity(%)": 70.0,
  "Pressure(in)": 29.9,
  "Visibility(mi)": 10.0,
  "Wind_Speed(mph)": 10.0,
  "Precipitation(in)": 0.0,
  "model_name": null
}
```

## POST /predict – Response

```json
{
  "prediction": 1,
  "severity_label": "High",
  "probability": 0.7823,
  "model": "xgboost"
}
```
