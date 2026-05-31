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
    
    best_models = {}
    all_metrics = {}
    shap_importances = {}
    
    # We train a model for each forecasting target (1-day, 2-day, 3-day ahead)
    for horizon, target_col in TARGET_COLS.items():
        print(f"\n--- Training Models for Horizon: {horizon} ({target_col}) ---")
        
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        
        # Define candidate models
        candidates = {
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        best_r2 = -float("inf")
        best_model_name = ""
        best_model = None
        best_model_metrics = {}
        
        for name, model in candidates.items():
            # Fit
            model.fit(X_train, y_train)
            
            # Predict
            preds = model.predict(X_test)
            
            # Evaluate
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            print(f"  {name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
            
            # Selection Criteria: Highest R-squared
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model = model
                best_model_metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
                
        print(f"Selected Best Model for {horizon}: {best_model_name} (R2={best_r2:.2f})")
        best_models[horizon] = best_model
        all_metrics[horizon] = {
            "model_name": best_model_name,
            **best_model_metrics
