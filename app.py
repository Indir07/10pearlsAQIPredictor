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


def get_aqi_health_details(aqi: float) -> dict:
    """Returns color codes and health advisories based on US EPA AQI thresholds."""
    if aqi <= 50:
        return {
            "label": "Good",
            "color": "#10b981",  # Emerald Green
            "text_color": "#ffffff",
            "desc": "Air quality is satisfactory, and air pollution poses little or no risk.",
            "advisory": "Excellent day to spend time outdoors! Open windows to ventilate your home."
        }
    elif aqi <= 100:
        return {
            "label": "Moderate",
            "color": "#f59e0b",  # Amber Yellow
            "text_color": "#ffffff",
            "desc": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.",
            "advisory": "Sensitive individuals should consider reducing intense outdoor activities."
        }
    elif aqi <= 150:
        return {
            "label": "Unhealthy for Sensitive Groups",
            "color": "#f97316",  # Orange
            "text_color": "#ffffff",
            "desc": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
            "advisory": "People with respiratory/heart issues, children, and elderly should limit prolonged outdoor exertion."
        }
    elif aqi <= 200:
        return {
            "label": "Unhealthy",
            "color": "#ef4444",  # Red
            "text_color": "#ffffff",
            "desc": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.",
            "advisory": "Avoid prolonged outdoor exertion. Consider wearing a high-quality protective mask (N95) outdoors."
        }
    elif aqi <= 300:
        return {
            "label": "Very Unhealthy",
            "color": "#8b5cf6",  # Purple
            "text_color": "#ffffff",
            "desc": "Health alert: The risk of health effects is increased for everyone.",
            "advisory": "Restrict all outdoor activities. Keep doors and windows closed. Run air purifiers indoors."
        }
    else:
        return {
            "label": "Hazardous",
            "color": "#7f1d1d",  # Maroon
            "text_color": "#ffffff",
            "desc": "Health warning of emergency conditions: The entire population is more likely to be affected.",
            "advisory": "STAY INDOORS. Avoid any physical activity. Use recirculated air in HVAC systems."
        }

