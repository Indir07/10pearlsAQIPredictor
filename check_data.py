import pandas as pd
from config import FeatureStoreAdapter

def audit_feature_store():
    print("=== Auditing local SQLite Feature Store ===")
    
    # Read features
    df = FeatureStoreAdapter.read_features("aqi_features")
    
    if df.empty:
        print("[ERROR] Feature Store is empty. Run backfill.py first!")
        return
        
    print(f"[SUCCESS] Connection successful!")
    print(f"Total Records: {len(df)} hourly observations")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print("\n--- Feature Ingestion Summary ---")
    metrics = ["us_aqi", "pm2_5", "pm10", "ozone", "carbon_monoxide"]
    
    summary = []
    for m in metrics:
        if m in df.columns:
            summary.append({
                "Pollutant/Metric": m,
                "Min": df[m].min(),
                "Max": df[m].max(),
                "Average": df[m].mean()
            })
            
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    print("\n--- Training Targets Ingestion Summary ---")
    targets = ["target_aqi_1d", "target_aqi_2d", "target_aqi_3d"]
    target_summary = []
    for t in targets:
        if t in df.columns:
            # Note: targets might have some NaNs at the very end
            valid_count = df[t].notna().sum()
            target_summary.append({
                "Target Horizon": t,
                "Valid Training Labels": valid_count,
                "Average Future AQI": df[t].mean()
            })
    target_df = pd.DataFrame(target_summary)
    print(target_df.to_string(index=False))

if __name__ == "__main__":
    audit_feature_store()
