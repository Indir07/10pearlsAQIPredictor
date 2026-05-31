import pandas as pd
import numpy as np
from datetime import datetime
from data_loader import fetch_aqi_data
from config import FeatureStoreAdapter, DEFAULT_CITY_NAME, DEFAULT_LATITUDE, DEFAULT_LONGITUDE


def compute_features(df: pd.DataFrame, include_targets: bool = False) -> pd.DataFrame:
    """
    Computes time-based, rolling-window, and derived features from raw hourly data.
    
    Parameters:
        df (pd.DataFrame): Raw DataFrame from data loader.
        include_targets (bool): If True, computes and includes target labels for training.
        
    Returns:
        pd.DataFrame: Engineered feature DataFrame.
    """
    if df.empty:
        return pd.DataFrame()
        
    # Ensure correct sorting by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 1. Time-Based Features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    
    # 2. Rolling Window Features (6h and 24h means)
    for col in ["pm2_5", "pm10", "us_aqi"]:
        df[f"{col}_roll_6h"] = df[col].rolling(window=6, min_periods=1).mean()
        df[f"{col}_roll_24h"] = df[col].rolling(window=24, min_periods=1).mean()
        
    # 3. Derived Features: AQI change rates
    # Current AQI minus AQI 6 hours ago
    df["aqi_change_rate_6h"] = df["us_aqi"] - df["us_aqi"].shift(6)
    # Current AQI minus AQI 24 hours ago
    df["aqi_change_rate_24h"] = df["us_aqi"] - df["us_aqi"].shift(24)
    
