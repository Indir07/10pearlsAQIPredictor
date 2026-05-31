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

