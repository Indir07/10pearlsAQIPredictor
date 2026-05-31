import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import FeatureStoreAdapter, MODEL_DIR

# Defined Feature columns
FEATURE_COLS = [
    "hour", "day_of_week", "month",
    "pm2_5", "pm10", "us_aqi",
    "pm2_5_roll_6h", "pm2_5_roll_24h",
    "pm10_roll_6h", "pm10_roll_24h",
    "us_aqi_roll_6h", "us_aqi_roll_24h",
    "aqi_change_rate_6h", "aqi_change_rate_24h"
]

TARGET_COLS = {
    "1d": "target_aqi_1d",
    "2d": "target_aqi_2d",
    "3d": "target_aqi_3d"
}


def run_training_pipeline():
    print("=== Starting Model Training Pipeline ===")
    
    # 1. Fetch historical features and targets from the Feature Store
    df = FeatureStoreAdapter.read_features("aqi_features")
    if df.empty:
        print("Error: Feature Store is empty. Have you run backfill.py yet?")
        return
        
    print(f"Loaded {len(df)} records from the Feature Store.")
    
    # Ensure sorted by timestamp for temporal splitting
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Drop rows containing NaNs in either features or targets
    full_cols = FEATURE_COLS + list(TARGET_COLS.values())
    df = df.dropna(subset=full_cols)
    
    if len(df) < 100:
        print(f"Warning: Only {len(df)} fully complete records found. Need more data to train models.")
        return
        
    # 2. Temporal Split (80% Train, 20% Test) to respect time sequence
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Training on first {len(train_df)} rows. Testing on remaining {len(test_df)} rows.")
    
    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]
