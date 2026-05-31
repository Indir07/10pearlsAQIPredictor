import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import FeatureStoreAdapter, MODEL_DIR

# Expanded Feature columns with cyclical encoding
FEATURE_COLS = [
    "hour", "day_of_week", "month",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
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
    
    # 5-Fold Time-Series Cross-Validation setup
    tscv = TimeSeriesSplit(n_splits=5)
    
    # We train a model for each forecasting target (1-day, 2-day, and 3-day ahead)
    for horizon, target_col in TARGET_COLS.items():
        print(f"\n--- Training Models for Horizon: {horizon} ({target_col}) ---")
        
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        
        # Calculate Naive Persistence Baseline (predict future AQI = current AQI)
        y_pred_naive = X_test["us_aqi"]
        naive_rmse = np.sqrt(mean_squared_error(y_test, y_pred_naive))
        naive_mae = mean_absolute_error(y_test, y_pred_naive)
        naive_r2 = r2_score(y_test, y_pred_naive)
        
        print(f"  [Naive Baseline Comparison] Persistence Forecast:")
        print(f"    RMSE = {naive_rmse:.2f}, MAE = {naive_mae:.2f}, R² = {naive_r2:.2f}")
        
        # 1. Hyperparameter Tuning for Ridge Regression using Pipeline to prevent data leakage
        print("  Running GridSearchCV for Ridge Regression Pipeline...")
        ridge_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge())
        ])
        ridge_cv = GridSearchCV(
            ridge_pipe,
            {"regressor__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
            cv=tscv,
            scoring="neg_mean_squared_error"
        )
        ridge_cv.fit(X_train, y_train)
        best_ridge_alpha = ridge_cv.best_params_["regressor__alpha"]
        print(f"    -> Best Ridge Alpha: {best_ridge_alpha}")
        
        # 2. Hyperparameter Tuning for HistGradientBoosting (LightGBM equivalent) Pipeline
        print("  Running GridSearchCV for HistGradientBoosting (LightGBM) Pipeline...")
        hgb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", HistGradientBoostingRegressor(random_state=42))
        ])
        hgb_cv = GridSearchCV(
            hgb_pipe,
            {
                "regressor__max_depth": [3, 5, 8],
                "regressor__learning_rate": [0.01, 0.05, 0.1],
                "regressor__max_iter": [50, 100]
            },
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        hgb_cv.fit(X_train, y_train)
        best_hgb_params = {
            k.replace("regressor__", ""): v for k, v in hgb_cv.best_params_.items()
        }
        print(f"    -> Best HistGradientBoosting Params: {best_hgb_params}")
        
        # Define candidate pipeline models with best tuned parameters
        tuned_ridge = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=best_ridge_alpha))
        ])
        
        tuned_hgb = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", HistGradientBoostingRegressor(random_state=42, **best_hgb_params))
        ])
        
        rf_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
        ])
        
        # Combined VotingRegressor Ensemble Pipeline
        ensemble_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", VotingRegressor(estimators=[
                ("ridge", Ridge(alpha=best_ridge_alpha)),
                ("hgb", HistGradientBoostingRegressor(random_state=42, **best_hgb_params)),
                ("rf", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
            ]))
        ])
        
        candidates = {
            "Tuned Ridge Regression": tuned_ridge,
            "Tuned HistGradientBoosting (LightGBM)": tuned_hgb,
            "Random Forest": rf_pipe,
            "Voting Regressor Ensemble": ensemble_pipe
        }
        
        best_r2 = -float("inf")
        best_model_name = ""
        best_model = None
        best_model_metrics = {}
        
        # Evaluate candidate models on the held-out test split
        for name, model in candidates.items():
            # Explicitly fit the model pipeline on full training set first
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            test_mae = mean_absolute_error(y_test, test_preds)
            test_r2 = r2_score(y_test, test_preds)
            
            print(f"  {name} Test Metrics: RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R²={test_r2:.2f}")
            
            # Selection Criteria: Highest R-squared on validation test set
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model_name = name
                best_model = model
                best_model_metrics = {"rmse": float(test_rmse), "mae": float(test_mae), "r2": float(test_r2)}
                
        print(f"Selected Best Model for {horizon}: {best_model_name} (Test R² = {best_r2:.2f})")
        best_models[horizon] = best_model
        all_metrics[horizon] = {
            "model_name": best_model_name,
            **best_model_metrics,
            "naive_comparison": {
                "rmse": float(naive_rmse),
                "mae": float(naive_mae),
                "r2": float(naive_r2)
            }
        }
        
        # 3. Calculate SHAP Explanations for the best model
        try:
            print(f"  Calculating SHAP explanations for {best_model_name}...")
            sample_X = X_test.sample(min(100, len(X_test)), random_state=42)
            
            # Extract preprocessing and final regressor
            preprocessing_steps = best_model.steps[:-1]
            if preprocessing_steps:
                preprocessor = Pipeline(preprocessing_steps)
                sample_X_preprocessed = preprocessor.transform(sample_X)
                # Re-wrap in DataFrame to keep column headers for SHAP
                sample_X_preprocessed = pd.DataFrame(sample_X_preprocessed, columns=FEATURE_COLS, index=sample_X.index)
            else:
                sample_X_preprocessed = sample_X
                
            regressor = best_model.named_steps["regressor"]
            
            explain_model = regressor
            if isinstance(regressor, VotingRegressor):
                print("  Best model is an Ensemble. Using the HistGradientBoosting base model for SHAP explanations.")
                explain_model = regressor.named_estimators_["hgb"]
            
            # Define explainer
            if isinstance(explain_model, (RandomForestRegressor, HistGradientBoostingRegressor)):
                explainer = shap.TreeExplainer(explain_model)
                shap_values = explainer.shap_values(sample_X_preprocessed)
            else:
                # Linear/Kernel Explainer for Ridge
                explainer = shap.Explainer(explain_model.predict, sample_X_preprocessed)
                shap_values = explainer(sample_X_preprocessed).values
                
            if isinstance(shap_values, list):
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
                
            # Create feature ranking dictionary
            feat_imp = dict(zip(FEATURE_COLS, [float(x) for x in mean_shap]))
            feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
            shap_importances[horizon] = feat_imp
            print(f"  SHAP Calculations successful. Top feature: {list(feat_imp.keys())[0]}")
        except Exception as e:
            print(f"  Warning: Failed to calculate SHAP: {e}")
            shap_importances[horizon] = {}

    # Pack models and precalculated SHAP importances
    payload = {
        "models": best_models,
        "shap_importances": shap_importances,
        "feature_cols": FEATURE_COLS
    }
    
    # 4. Save models and metrics to Model Registry
    FeatureStoreAdapter.save_model(
        models=payload,
        model_name="aqi_prediction_models",
        metrics=all_metrics
    )
    
    print("\n=== Model Training & Registry Run Completed Successfully! ===")


if __name__ == "__main__":
    run_training_pipeline()
