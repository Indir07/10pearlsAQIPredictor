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

        # Ensure timestamp column exists and is cast properly
        if "timestamp" in df.columns:
            # Convert to string format if necessary for serialization
            df = df.copy()
            df["timestamp"] = df["timestamp"].astype(str)

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
                        online_enabled=True
                    )
                    print(f"Inserting {len(df)} records into Hopsworks Feature Group '{fg_name}'...")
                    fg.insert(df, write_options={"wait_for_job": False})
                    print("Insertion completed successfully.")
                    return
            except Exception as e:
                print(f"Failed to write to Hopsworks: {e}. Writing to Local SQLite instead.")

        # Local SQLite Fallback
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
