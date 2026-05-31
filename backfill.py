import pandas as pd
from datetime import datetime, timedelta
from data_loader import fetch_aqi_data
from feature_pipeline import compute_features
from config import FeatureStoreAdapter, DEFAULT_CITY_NAME, DEFAULT_LATITUDE, DEFAULT_LONGITUDE


def backfill_historical_data(days_of_history: int = 365):
    """
    Fetches and prepares historical features and targets to backfill the Feature Store.
    
    Parameters:
        days_of_history (int): Number of past days of hourly data to retrieve (default 365).
    """
    print(f"=== Starting Historical Feature Store Backfill for {DEFAULT_CITY_NAME} ===")
    
    # Calculate dates
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_of_history)
    
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")
    
    # 1. Fetch raw historical data
    raw_df = fetch_aqi_data(
        latitude=DEFAULT_LATITUDE,
        longitude=DEFAULT_LONGITUDE,
        start_date=start_date,
        end_date=end_date
    )
    
    if raw_df.empty:
        print("Error: Historical raw data is empty. Backfill aborted.")
        return
        
    print(f"Fetched {len(raw_df)} raw hourly rows. Running feature engineering & target generation...")
    
    # 2. Compute features and target variables
    # For training data, we MUST include targets (future values) and drop incomplete rows
    train_df = compute_features(raw_df, include_targets=True)
    
    if train_df.empty:
        print("Error: Feature engineering yielded empty training set. Backfill aborted.")
        return
        
    print(f"Feature engineering complete. {len(train_df)} rows are fully formed with historical features and targets.")
    
    # 3. Store processed features into the Feature Store
    FeatureStoreAdapter.insert_features(train_df, group_name="aqi_features")
    print("=== Backfill Process Completed Successfully! ===")


if __name__ == "__main__":
    # Run backfill
    backfill_historical_data(days_of_history=365)
