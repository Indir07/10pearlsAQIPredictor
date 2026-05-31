import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data_loader import fetch_aqi_data
from feature_pipeline import compute_features
from config import FeatureStoreAdapter, DEFAULT_CITY_NAME, DEFAULT_LATITUDE, DEFAULT_LONGITUDE

# Page Configuration
st.set_page_config(
    page_title="Pearls AQI Forecast Dashboard",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-card {
        border-radius: 16px;
        padding: 24px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .aqi-val {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
        margin: 10px 0;
    }
    
    .aqi-label {
        font-size: 1.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)
