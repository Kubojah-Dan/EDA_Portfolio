from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text
from sqlalchemy.sql import func
from app.backend.database import Base


class Accident(Base):
    __tablename__ = "accidents"

    id = Column(Integer, primary_key=True, index=True)
    accident_id = Column(String(20), unique=True, index=True)
    severity = Column(Integer, nullable=False, index=True)
    start_time = Column(DateTime, index=True)
    start_lat = Column(Float)
    start_lng = Column(Float)
    city = Column(String(100), index=True)
    state = Column(String(5), index=True)
    county = Column(String(100))
    zipcode = Column(String(20))
    temperature_f = Column(Float)
    humidity_pct = Column(Float)
    visibility_mi = Column(Float)
    wind_speed_mph = Column(Float)
    precipitation_in = Column(Float)
    weather_condition = Column(String(100))
    distance_mi = Column(Float)
    sunrise_sunset = Column(String(10))
    junction = Column(Boolean, default=False)
    traffic_signal = Column(Boolean, default=False)
    crossing = Column(Boolean, default=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), index=True)
    test_auc = Column(Float)
    test_f1 = Column(Float)
    test_precision = Column(Float)
    test_recall = Column(Float)
    test_accuracy = Column(Float)
    test_ap = Column(Float)
    n_train = Column(Integer)
    n_test = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50))
    prediction = Column(Integer)
    probability = Column(Float)
    severity_label = Column(String(10))
    input_features = Column(Text)  # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())

