# Pearls AQI Predictor 💨

An end-to-end, 100% serverless machine learning pipeline that forecasts the **Air Quality Index (AQI) up to 3 days in advance** for any city globally.

---

## 🏗️ Architecture Overview

This project implements a fully productionized ML pipeline containing data ingestion, rolling feature engineering, historical backfilling, multi-horizon model training, explainability, and a dynamic real-time Streamlit dashboard:

1. **Feature Ingestion & Pipeline**: Automatically downloads weather and pollutant metrics from the free **Open-Meteo Air Quality API** (completely open, no key registration needed).
2. **Dual-Mode Feature Store**:
   - **Hopsworks Integration (Cloud Mode)**: Writes features and retrieves them serverlessly in the cloud if an API key is configured.
   - **SQLite Adapter (Local Mode)**: Stores and reads data from a local SQLite database (`local_feature_store.db`) if no key is set. Works **100% out of the box!**
3. **Multi-Horizon Training Pipeline**:
   - Compares Ridge Regression, Random Forest, and Gradient Boosting Regressors.
   - Evaluates performance using standard regressions metrics: RMSE, MAE, and $R^2$.
   - Pre-computes **SHAP (Shapley Additive exPlanations)** values to identify feature importances.
4. **Dynamic Streamlit Web App**:
   - Shows the current AQI with EPA-defined color codings and health advisories.
   - Compares the baseline forecast with our ML predictions on interactive Plotly graphs.
   - Visualizes SHAP feature importances.
5. **CI/CD Orchestration**:
   - GitHub Actions workflow (`pipeline.yml`) runs the feature pipeline hourly and the model training pipeline daily.

---

## 📁 Repository Structure

```
aqi_predictor/
│
├── .github/
│   └── workflows/
│       └── pipeline.yml       # GitHub Actions workflow for schedules
│
├── data/                      # Local storage for databases and model pickles
│   ├── local_feature_store.db # SQLite feature store
│   └── models/                # Saved trained model pickles and metadata
│
├── .env                       # Local environment variables
├── requirements.txt           # Python dependencies
├── config.py                  # Dual-store database adapter & configuration loader
├── data_loader.py             # Open-Meteo API ingestion client
├── feature_pipeline.py        # Hourly feature processor
├── backfill.py                # Script to populate Feature Store with historical data
├── training_pipeline.py       # Model trainer, evaluator, and SHAP logger
└── app.py                     # Premium Streamlit web application
```

---

## ⚡ Getting Started (Local Run)

### 1. Clone & Setup Workspace
Navigate to your repository and create your workspace.

### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then install the package requirements:
```bash
pip install -r requirements.txt
```

### 3. Local Environment File (`.env`)
The project comes pre-configured with **New York City** as the default. To change this or add your Hopsworks credentials, edit the `.env` file:
```env
HOPSWORKS_API_KEY=your-api-key-here # (Optional) Leave empty to use SQLite mode!
DEFAULT_CITY_NAME=New York
DEFAULT_LATITUDE=40.7128
DEFAULT_LONGITUDE=-74.0060
```

### 4. Backfill Historical Data
Populate your feature store with 365 days of hourly air quality records:
```bash
python backfill.py
```

### 5. Train & Evaluate Models
Run the training pipeline to evaluate the ML candidates, compute SHAP importances, and register the best models:
```bash
python training_pipeline.py
```

### 6. Launch the Premium Dashboard
Start the interactive dashboard locally:
```bash
streamlit run app.py
```

---

## 🤖 Cloud Setup: Hopsworks Feature Store

If you would like to run this in a true serverless cloud architecture:
1. Register for a free account at [Hopsworks.ai](https://www.hopsworks.ai/).
2. Create a project and navigate to **Settings > Api Keys** to generate an API key with all permissions checked.
3. Paste the key into your `.env` file (`HOPSWORKS_API_KEY=your-key`).
4. Re-run `backfill.py` and `training_pipeline.py`. The scripts will automatically detect the key, connect to Hopsworks, create cloud feature groups, and save models directly to the cloud Model Registry!

---

## 🚀 CI/CD Automation: GitHub Actions

To automate your pipelines serverlessly on a schedule:
1. Commit the codebase to your GitHub Repository.
2. In your GitHub Repository, go to **Settings > Secrets and variables > Actions**.
3. Create a repository secret named `HOPSWORKS_API_KEY` and paste your key.
4. The workflow will automatically trigger **every hour** to pull live AQI data, and **every night** to retrain your forecasting models.
