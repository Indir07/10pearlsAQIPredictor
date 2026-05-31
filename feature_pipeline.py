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
    
    # Cyclical temporal encoding (enables models to see continuity, e.g. Hour 23 adjacent to Hour 0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    
    # 2. Rolling Window Features (6h and 24h means)
    for col in ["pm2_5", "pm10", "us_aqi"]:
        df[f"{col}_roll_6h"] = df[col].rolling(window=6, min_periods=1).mean()
        df[f"{col}_roll_24h"] = df[col].rolling(window=24, min_periods=1).mean()
        
    # 3. Derived Features: AQI change rates
    # Current AQI minus AQI 6 hours ago
    df["aqi_change_rate_6h"] = df["us_aqi"] - df["us_aqi"].shift(6)
    # Current AQI minus AQI 24 hours ago
    df["aqi_change_rate_24h"] = df["us_aqi"] - df["us_aqi"].shift(24)
    
    # Fill shifting NaNs with 0 or forward fill to prevent issues
    df["aqi_change_rate_6h"] = df["aqi_change_rate_6h"].fillna(0)
    df["aqi_change_rate_24h"] = df["aqi_change_rate_24h"].fillna(0)
    
    # 4. Targets (Future US AQI predictions: 1-day, 2-day, and 3-day ahead)
    if include_targets:
        # target_aqi_1d = US AQI 24 hours into the future
        df["target_aqi_1d"] = df["us_aqi"].shift(-24)
        # target_aqi_2d = US AQI 48 hours into the future
        df["target_aqi_2d"] = df["us_aqi"].shift(-48)
        # target_aqi_3d = US AQI 72 hours into the future
        df["target_aqi_3d"] = df["us_aqi"].shift(-72)
        
        # Drop rows where target labels are NaN (since they are in the future and cannot be used to train)
        df = df.dropna(subset=["target_aqi_1d", "target_aqi_2d", "target_aqi_3d"])
        
    return df


def run_hourly_feature_pipeline():
    """
    Retrieves the latest 72 hours of data (to capture rolling history),
    computes current features, and inserts them into the feature store.
    """
    print(f"[{datetime.now()}] Running Hourly Feature Pipeline...")
    
    # Fetch recent data
    raw_df = fetch_aqi_data(DEFAULT_LATITUDE, DEFAULT_LONGITUDE)
    if raw_df.empty:
        print("Error: Could not retrieve raw data. Feature pipeline aborted.")
        return
        
    # We do NOT include targets during hourly prediction runs, as we are writing
    # the latest feature states to be consumed by the dashboard for live predictions.
    featured_df = compute_features(raw_df, include_targets=False)
    
    # We only write the latest forecast rows (where current data is active)
    # Open-Meteo returns past 2 days (48 rows) and future 7 days (168 rows).
    # Let's save all rows into our feature store so we keep a running history of hourly features.
    FeatureStoreAdapter.insert_features(featured_df, group_name="aqi_features")
    print("Hourly Feature Pipeline run completed successfully.")


if __name__ == "__main__":
    run_hourly_feature_pipeline()
