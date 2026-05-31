import os
import sqlite3
import pickle
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Feature Store Details
SQLITE_DB_PATH = DATA_DIR / "local_feature_store.db"

# Default Configuration Settings
DEFAULT_CITY_NAME = os.getenv("DEFAULT_CITY_NAME", "New York")
DEFAULT_LATITUDE = float(os.getenv("DEFAULT_LATITUDE", "40.7128"))
DEFAULT_LONGITUDE = float(os.getenv("DEFAULT_LONGITUDE", " -74.0060"))

# Hopsworks Setup
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "").strip()
try:
    import streamlit as st
    if "HOPSWORKS_API_KEY" in st.secrets:
        HOPSWORKS_API_KEY = st.secrets["HOPSWORKS_API_KEY"].strip()
except Exception:
    pass

IS_HOPSWORKS_ENABLED = bool(HOPSWORKS_API_KEY)


def get_db_connection():
    """Returns a SQLite connection for the local feature store fallback."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    return conn


class FeatureStoreAdapter:
    """Unified adapter interface for storing and retrieving features and models."""
    
    @staticmethod
    def is_using_hopsworks():
        return IS_HOPSWORKS_ENABLED

    @classmethod
    def get_feature_store(cls):
        """Connects to Hopsworks Feature Store or returns 'local'."""
        if IS_HOPSWORKS_ENABLED:
            import hopsworks
            try:
                print("Connecting to Hopsworks cloud feature store...")
                project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
                return project.get_feature_store()
            except Exception as e:
                print(f"Error connecting to Hopsworks: {e}. Falling back to Local Store.")
        return "local"

    @classmethod
    def insert_features(cls, df: pd.DataFrame, group_name: str = "aqi_features"):
        """Inserts a pandas DataFrame into the Feature Store."""
        if df.empty:
            print("DataFrame is empty, skipping insertion.")
            return

        # Ensure timestamp column exists and is parsed as datetime
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        if IS_HOPSWORKS_ENABLED:
            try:
                fs = cls.get_feature_store()
                if fs != "local":
                    # Hopsworks feature group names must be lowercase
                    fg_name = group_name.lower()
                    fg = fs.get_or_create_feature_group(
                        name=fg_name,
                        version=1,
                        primary_key=["timestamp"],
                        event_time="timestamp",
                        description="AQI forecast and weather features",
                        online_enabled=False
                    )
                    print(f"Inserting {len(df)} records into Hopsworks Feature Group '{fg_name}'...")
                    fg.insert(df, write_options={"wait_for_job": False})
                    print("Insertion completed successfully.")
                    return
            except Exception as e:
                print(f"Failed to write to Hopsworks: {e}. Writing to Local SQLite instead.")

        # Local SQLite Fallback
        if "timestamp" in df.columns:
            # Convert to string format for SQLite serialization
            df["timestamp"] = df["timestamp"].astype(str)

        print(f"Inserting {len(df)} records into Local SQLite Table '{group_name}'...")
        conn = get_db_connection()
        try:
            # We want to replace existing records on primary key conflict if possible
            # SQLite does not easily support REPLACE with to_sql directly unless we write SQL or drop/replace.
            # A clean way is to load existing records, drop duplicates, and rewrite, or insert OR REPLACE.
            # To keep it extremely simple, we write to_sql with append, then run a deduplication query.
            df.to_sql(group_name, conn, if_exists="append", index=False)
            
            # Deduplicate by timestamp
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS temp_{group_name} AS 
                SELECT * FROM {group_name} GROUP BY timestamp;
            """)
            cursor.execute(f"DROP TABLE {group_name};")
            cursor.execute(f"ALTER TABLE temp_{group_name} RENAME TO {group_name};")
            conn.commit()
            print("Insertion completed successfully (Local SQLite).")
        except Exception as e:
            print(f"SQLite Write Error: {e}")
        finally:
            conn.close()

    @classmethod
    def read_features(cls, group_name: str = "aqi_features") -> pd.DataFrame:
        """Reads features from the Feature Store and returns a pandas DataFrame."""
        if IS_HOPSWORKS_ENABLED:
            try:
                fs = cls.get_feature_store()
                if fs != "local":
                    fg_name = group_name.lower()
                    fg = fs.get_feature_group(name=fg_name, version=1)
                    print(f"Reading from Hopsworks Feature Group '{fg_name}'...")
                    df = fg.read()
                    # Parse timestamp back to datetime
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    return df
            except Exception as e:
                print(f"Failed to read from Hopsworks: {e}. Reading from Local SQLite instead.")

        # Local SQLite Fallback
        print(f"Reading from Local SQLite Table '{group_name}'...")
        conn = get_db_connection()
        try:
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{group_name}';")
            if not cursor.fetchone():
                print(f"Table '{group_name}' does not exist yet. Returning empty DataFrame.")
                return pd.DataFrame()
                
            df = pd.read_sql_query(f"SELECT * FROM {group_name} ORDER BY timestamp ASC", conn)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            print(f"SQLite Read Error: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    @classmethod
    def save_model(cls, models: dict, model_name: str = "aqi_prediction_models", metrics: dict = None):
        """Saves models and metrics to local files and/or Hopsworks Model Registry."""
        # 1. Local Saving
        local_path = MODEL_DIR / f"{model_name}.pkl"
        meta_path = MODEL_DIR / f"{model_name}_metadata.pkl"
        
        with open(local_path, "wb") as f:
            pickle.dump(models, f)
            
        metadata = {"metrics": metrics or {}}
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
            
        print(f"Successfully saved models locally to {local_path}")

        # 2. Hopsworks Model Registry Saving
        if IS_HOPSWORKS_ENABLED:
            try:
                import hopsworks
                print("Connecting to Hopsworks Model Registry...")
                project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
                mr = project.get_model_registry()
                
                # Flat, numeric metrics for Hopsworks Model Registry
                hw_metrics = {}
                if metrics:
                    for horizon, h_metrics in metrics.items():
                        for k, v in h_metrics.items():
                            if isinstance(v, (int, float)):
                                hw_metrics[f"{horizon}_{k}"] = float(v)
                            elif k == "naive_comparison" and isinstance(v, dict):
                                for nk, nv in v.items():
                                    if isinstance(nv, (int, float)):
                                        hw_metrics[f"{horizon}_naive_{nk}"] = float(nv)

                # Create Model in Registry
                hw_model = mr.python.create_model(
                    name=model_name,
                    metrics=hw_metrics,
                    description="Random Forest & Ridge regressors for 1-day, 2-day, and 3-day AQI prediction."
                )
                
                # Save the model artifact to registry
                hw_model.save(str(local_path))
                print(f"Successfully registered model '{model_name}' on Hopsworks Model Registry!")
            except Exception as e:
                print(f"Failed to register model on Hopsworks: {e}. Model is saved locally.")

    @classmethod
    def load_model(cls, model_name: str = "aqi_prediction_models") -> tuple:
        """Loads models and metrics from Hopsworks or Local storage."""
        # Check Hopsworks registry if enabled
        if IS_HOPSWORKS_ENABLED:
            try:
                import hopsworks
                print("Attempting to load model from Hopsworks Model Registry...")
                project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
                mr = project.get_model_registry()
                
                hw_model = mr.get_model(model_name, version=1)
                model_dir = hw_model.download()
                
                model_path = Path(model_dir) / f"{model_name}.pkl"
                with open(model_path, "rb") as f:
                    models = pickle.load(f)
                    
                # In Hopsworks python SDK, metrics are stored in the training_metrics attribute of the Model object
                raw_metrics = getattr(hw_model, "training_metrics", {}) or {}
                
                # Reconstruct the nested metrics dictionary for the Streamlit dashboard
                metrics = {}
                if raw_metrics:
                    for horizon in ["1d", "2d", "3d"]:
                        metrics[horizon] = {
                            "model_name": "Tuned HistGradientBoosting (LightGBM)",
                            "rmse": float(raw_metrics.get(f"{horizon}_rmse", 0.0)),
                            "mae": float(raw_metrics.get(f"{horizon}_mae", 0.0)),
                            "r2": float(raw_metrics.get(f"{horizon}_r2", 0.0)),
                            "naive_comparison": {
                                "rmse": float(raw_metrics.get(f"{horizon}_naive_rmse", 0.0)),
                                "mae": float(raw_metrics.get(f"{horizon}_naive_mae", 0.0)),
                                "r2": float(raw_metrics.get(f"{horizon}_naive_r2", 0.0))
                            }
                        }
                    
                print("Model loaded successfully from Hopsworks.")
                return models, metrics
            except Exception as e:
                print(f"Failed to load from Hopsworks: {e}. Attempting local load...")
                return None, f"Hopsworks Error: {str(e)}"

        # Local Load Fallback
        local_path = MODEL_DIR / f"{model_name}.pkl"
        meta_path = MODEL_DIR / f"{model_name}_metadata.pkl"
        
        if not local_path.exists():
            print(f"No trained model found at {local_path}.")
            # Return any collected error message if present
            return None, locals().get("e", None)
            
        with open(local_path, "rb") as f:
            models = pickle.load(f)
            
        metrics = {}
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                metrics = meta.get("metrics", {})
                
        print("Model loaded successfully from Local Storage.")
        return models, metrics
