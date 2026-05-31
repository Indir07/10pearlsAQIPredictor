import requests
import pandas as pd
from datetime import datetime, timedelta

# API endpoint
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_aqi_data(
    latitude: float,
    longitude: float,
    start_date: str = None,
    end_date: str = None,
    timezone: str = "auto"
) -> pd.DataFrame:
    """
    Fetches raw weather and pollutant data from Open-Meteo Air Quality API.
    
    Parameters:
        latitude (float): Latitude of the target location.
        longitude (float): Longitude of the target location.
        start_date (str, optional): Start date in YYYY-MM-DD format.
        end_date (str, optional): End date in YYYY-MM-DD format.
        timezone (str): Timezone setting (default "auto").
        
    Returns:
        pd.DataFrame: DataFrame containing hourly weather and pollutant features.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "us_aqi,pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide",
        "timezone": timezone
    }
    
    # Add date range if specified (for backfilling historical data)
    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
        print(f"Fetching historical AQI data for ({latitude}, {longitude}) from {start_date} to {end_date}...")
    else:
        # Defaults to past 2 days and next 7 days forecast if no dates are passed
        print(f"Fetching real-time & forecast AQI data for ({latitude}, {longitude})...")
        
    try:
        response = requests.get(OPEN_METEO_AIR_QUALITY_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if "hourly" not in data:
            print("API response does not contain 'hourly' data.")
            return pd.DataFrame()
            
        hourly_data = data["hourly"]
        
        # Construct DataFrame
        df = pd.DataFrame(hourly_data)
        
        # Rename 'time' to 'timestamp'
        df.rename(columns={"time": "timestamp"}, inplace=True)
        
        # Parse timestamp column
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Add metadata columns
        df["latitude"] = latitude
        df["longitude"] = longitude
        df["utc_offset_seconds"] = data.get("utc_offset_seconds", 0)
        
        print(f"Successfully fetched {len(df)} hourly air quality records.")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed while fetching AQI data: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while parsing AQI data: {e}")
        return pd.DataFrame()


# Simple testing block
if __name__ == "__main__":
    # Test NYC
    print("Testing data loader for New York City...")
    test_df = fetch_aqi_data(40.7128, -74.0060)
    if not test_df.empty:
        print("\nFirst 5 records:")
        print(test_df.head())
        print("\nColumns:", list(test_df.columns))
    else:
        print("Test failed: Received empty DataFrame.")
