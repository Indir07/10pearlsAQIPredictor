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


# Popular cities coordinates mapping
POPULAR_CITIES = {
    "Karachi (Pakistan)": {"lat": 24.8607, "lon": 67.0011},
    "Lahore (Pakistan)": {"lat": 31.5204, "lon": 74.3587},
    "Islamabad (Pakistan)": {"lat": 33.6844, "lon": 73.0479},
    "Peshawar (Pakistan)": {"lat": 33.9971, "lon": 71.5725},
    "New York (USA)": {"lat": 40.7128, "lon": -74.0060},
    "London (UK)": {"lat": 51.5074, "lon": -0.1278},
    "Tokyo (Japan)": {"lat": 35.6762, "lon": 139.6503},
    "Bangalore (India)": {"lat": 12.9716, "lon": 77.5946},
    "Sydney (Australia)": {"lat": -33.8688, "lon": 151.2093},
    "Paris (France)": {"lat": 48.8566, "lon": 2.3522},
    "Delhi (India)": {"lat": 28.6139, "lon": 77.2090},
}

# HEADER
st.title("💨 Pearls Air Quality Index (AQI) Predictor")
st.markdown("An end-to-end serverless ML pipeline predicting air quality indices 3 days into the future.")

# SIDEBAR: City Selector & Custom Coordinates
st.sidebar.header("🌍 Location Settings")
city_mode = st.sidebar.radio("Select Input Mode", ["Popular Cities", "Custom Coordinates"])

if city_mode == "Popular Cities":
    selected_city = st.sidebar.selectbox("Choose a City", list(POPULAR_CITIES.keys()))
    lat = POPULAR_CITIES[selected_city]["lat"]
    lon = POPULAR_CITIES[selected_city]["lon"]
    city_name = selected_city.split(" (")[0]
else:
    city_name = st.sidebar.text_input("City Name", "Custom Location")
    lat = st.sidebar.number_input("Latitude", value=DEFAULT_LATITUDE, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=DEFAULT_LONGITUDE, format="%.4f")

# Load models and metrics
models_payload, mr_metrics = FeatureStoreAdapter.load_model("aqi_prediction_models")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Registry Status")
if models_payload is not None:
    st.sidebar.success("✅ ML Models loaded successfully")
    st.sidebar.info(f"Feature Store: {'Hopsworks (Cloud)' if FeatureStoreAdapter.is_using_hopsworks() else 'SQLite (Local)'}")
    
    # Render metrics in sidebar
    st.sidebar.markdown("**Validation Performance (R²):**")
    for key, metric in mr_metrics.items():
        st.sidebar.write(f"- **{key} Ahead:** R² = {metric.get('r2', 0.0):.2f}")
else:
    st.sidebar.warning("⚠️ No trained models found in Registry.")
    st.sidebar.info("Using baseline Open-Meteo physical forecast models.")

# --- DATA RETRIEVAL ---
with st.spinner(f"Fetching and processing hourly air quality features for {city_name}..."):
    raw_df = fetch_aqi_data(lat, lon)

if raw_df.empty:
    st.error("❌ Failed to fetch data from Open-Meteo API. Please verify coordinates and internet connection.")
else:
    # Compute features for prediction
    featured_df = compute_features(raw_df, include_targets=False)
    
    # The API returns past data (usually 48 hours) and future data.
    # Current hour record is the transition point.
    now_utc = datetime.utcnow()
    # Find the row closest to the current local/system time
    featured_df["time_diff"] = (featured_df["timestamp"] - pd.Timestamp.now()).abs()
    current_idx = featured_df["time_diff"].idxmin()
    current_row = featured_df.loc[current_idx]
    
    # Current Metrics
    current_aqi = float(current_row["us_aqi"])
    health = get_aqi_health_details(current_aqi)
    
    # ------------------ HERO DISPLAY ------------------
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Beautiful dynamic CSS metric card
        st.markdown(f"""
        <div class="metric-card" style="background-color: {health['color']};">
            <span style="font-size: 0.9rem; text-transform: uppercase; font-weight: 600; opacity: 0.9;">Current AQI in {city_name}</span>
            <div class="aqi-val">{int(current_aqi)}</div>
            <div class="aqi-label">{health['label']}</div>
            <p style="margin-top: 15px; font-size: 0.95rem; opacity: 0.9; line-height: 1.4;">
                {health['desc']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display alert if forecast is hazardous
        if current_aqi > 100:
            st.warning(f"⚠️ **Health Advisory:** {health['advisory']}")
        else:
            st.info(f"💡 **Advisory:** {health['advisory']}")
            
    with col2:
        st.subheader("📊 Key Pollutant Concentration Details")
        st.markdown("Current hourly measures of critical pollutant concentrations (µg/m³ or index):")
        
        # Pollutant layout grid
        p_cols = st.columns(3)
        pollutants_map = {
            "PM2.5 (Fine Dust)": {"col": "pm2_5", "max": 75.0, "unit": " µg/m³"},
            "PM10 (Coarse Dust)": {"col": "pm10", "max": 150.0, "unit": " µg/m³"},
            "Ozone (O₃)": {"col": "ozone", "max": 180.0, "unit": " µg/m³"},
            "Nitrogen Dioxide (NO₂)": {"col": "nitrogen_dioxide", "max": 200.0, "unit": " µg/m³"},
            "Sulfur Dioxide (SO₂)": {"col": "sulphur_dioxide", "max": 350.0, "unit": " µg/m³"},
            "Carbon Monoxide (CO)": {"col": "carbon_monoxide", "max": 15000.0, "unit": " µg/m³"}
        }
        
        for idx, (label, spec) in enumerate(pollutants_map.items()):
            val = float(current_row[spec["col"]])
            # Progress value bounded 0 to 1
            progress_val = min(1.0, max(0.0, val / spec["max"]))
            
            p_cols[idx % 3].metric(label=label, value=f"{val:.1f}{spec['unit']}")
            p_cols[idx % 3].progress(progress_val)
            
    # ------------------ PREDICTIONS & FORECASTING ------------------
    st.markdown("---")
    st.subheader("📈 3-Day (72-Hour) AQI Forecasting Timeline")
    
    # We will compute forecast timelines:
    # 1. Physical Forecast (Open-Meteo raw baseline forecast)
    forecast_df = featured_df.loc[current_idx:].copy().head(73).reset_index(drop=True)
    
    # 2. Machine Learning Predictions (Multi-Horizon models)
    # Check if models exist. If yes, we predict 1d, 2d, 3d points
    ml_forecast = []
    
    if models_payload is not None:
        models = models_payload["models"]
        feat_cols = models_payload["feature_cols"]
        
        # We can predict at several points. Specifically:
        # At hour t, we can predict t+24h, t+48h, t+72h.
        # Let's map these exact dates:
        current_time = current_row["timestamp"]
        target_times = {
            "1d": current_time + timedelta(hours=24),
            "2d": current_time + timedelta(hours=48),
            "3d": current_time + timedelta(hours=72)
        }
        
        # Prepare input features vector from current time
        x_features = pd.DataFrame([current_row[feat_cols]], columns=feat_cols)
        
        # Make predictions
        for horizon in ["1d", "2d", "3d"]:
            pred_aqi = float(models[horizon].predict(x_features)[0])
            # Bounded prediction between 0 and 500
            pred_aqi = min(500.0, max(0.0, pred_aqi))
            ml_forecast.append({
                "horizon": f"{horizon.upper()} ML Forecast",
                "timestamp": target_times[horizon],
                "aqi": pred_aqi
            })
            
        ml_df = pd.DataFrame(ml_forecast)
        
    # --- PLOTLY INTERACTIVE CHART ---
    fig = go.Figure()
    
    # Historical Trend (past 24 hours)
    history_df = featured_df.loc[:current_idx].tail(24)
    fig.add_trace(go.Scatter(
        x=history_df["timestamp"],
        y=history_df["us_aqi"],
        name="Historical AQI (Past 24h)",
        line=dict(color="#4b5563", width=2, dash="dash"),
        mode="lines"
    ))
    
    # Open-Meteo Baseline Forecast (Next 72 hours)
    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["us_aqi"],
        name="Open-Meteo Physical Forecast",
        line=dict(color="#2563eb", width=3),
        mode="lines+markers"
    ))
    
    # ML Multi-Horizon Predictions
    if models_payload is not None:
        fig.add_trace(go.Scatter(
            x=ml_df["timestamp"],
            y=ml_df["aqi"],
            name="10Pearls ML Models Prediction",
            marker=dict(color="#ec4899", size=12, symbol="diamond"),
            mode="markers+text",
            text=[f"ML Predict: {int(x)}" for x in ml_df["aqi"]],
            textposition="top center",
            textfont=dict(family="Outfit", size=12, color="#ec4899")
        ))
        
    # Configure Chart Layout
    fig.update_layout(
        title=f"AQI Timeline & Model Projections for {city_name}",
        xaxis_title="Time / Date",
        yaxis_title="US Air Quality Index (AQI)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, max(200, featured_df["us_aqi"].max() + 50)])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast Hazard Warning Alert banner
    future_hazardous_points = []
    if models_payload is not None:
        for idx, row in ml_df.iterrows():
            if row["aqi"] > 100:
                health_level = get_aqi_health_details(row["aqi"])["label"]
                future_hazardous_points.append(f"**{row['horizon']}** ({row['timestamp'].strftime('%b %d, %H:%M')}): Predicted AQI **{int(row['aqi'])}** ({health_level})")
                
    if future_hazardous_points:
        st.error("🚨 **HAZARDOUS/UNHEALTHY AQI ELEVATION ALERT:**\n\n" + "\n\n".join(future_hazardous_points))
        
    # ------------------ FEATURE EXPLAINABILITY (SHAP) ------------------
    st.markdown("---")
    st.subheader("🔍 Machine Learning Model Interpretability (SHAP)")
    
    if models_payload is not None and "shap_importances" in models_payload:
        st.write("Understand *which* features have the most influence on the model predictions. The following shows the average absolute SHAP values (feature impact score):")
        
        horizon_to_show = st.selectbox("Select Horizon for SHAP Explanation", ["1d", "2d", "3d"])
        
        # Load SHAP importance dictionary
        shaps = models_payload["shap_importances"].get(horizon_to_show, {})
        
        if shaps:
            # Construct DataFrame for plotting
            shap_df = pd.DataFrame({
                "Feature": list(shaps.keys()),
                "Impact Score (SHAP value)": list(shaps.values())
            }).sort_values("Impact Score (SHAP value)", ascending=True)
            
            # Translate technical columns to friendly names
            friendly_names = {
                "us_aqi": "Current AQI Value",
                "pm2_5": "Current PM2.5",
                "pm10": "Current PM10",
                "us_aqi_roll_24h": "24h Rolling Average AQI",
                "us_aqi_roll_6h": "6h Rolling Average AQI",
                "pm2_5_roll_24h": "24h Rolling Average PM2.5",
                "pm2_5_roll_6h": "6h Rolling Average PM2.5",
                "pm10_roll_24h": "24h Rolling Average PM10",
                "pm10_roll_6h": "6h Rolling Average PM10",
                "aqi_change_rate_6h": "6-hour AQI Change Rate",
                "aqi_change_rate_24h": "24-hour AQI Change Rate",
                "hour": "Hour of Day",
                "day_of_week": "Day of Week",
                "month": "Month of Year"
            }
            shap_df["Friendly Feature"] = shap_df["Feature"].map(friendly_names).fillna(shap_df["Feature"])
            
            # Plot native horizontal bar chart
            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                y=shap_df["Friendly Feature"],
                x=shap_df["Impact Score (SHAP value)"],
                orientation="h",
                marker=dict(color="#10b981")
            ))
            
            fig_shap.update_layout(
                title=f"Feature Importances driving {horizon_to_show.upper()} predictions",
                xaxis_title="Average Impact |SHAP Value|",
                yaxis_title="Feature Name",
                template="plotly_white",
                height=450
            )
            
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP values not calculated for this model version.")
    else:
        st.info("💡 SHAP interpretability metrics will be available once the training pipeline has been executed at least once.")
