import streamlit as st
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

# --- Page Configuration ---
st.set_page_config(
    page_title="Noida Air Quality Forecast",
    page_icon="🌬️",
    layout="wide",
)

# --- Constants & Configuration ---
INPUT_WINDOW = 24 * 7
OUTPUT_WINDOW = 72
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
CITY = "Noida"
LAT, LON = 28.535517, 77.391029

# --- Caching Functions for Performance ---
@st.cache_resource
def load_keras_model():
    # Use the new .keras file name
    model = load_model('noida_pm25_transformer.keras')
    return model

@st.cache_data(ttl=3600) # Cache data for 1 hour to prevent re-downloading
def fetch_input_data():
    tz = pytz.timezone("Asia/Kolkata")
    start_date = (datetime.now(tz) - timedelta(days=8)).strftime('%Y-%m-%d')
    end_date = datetime.now(tz).strftime('%Y-%m-%d')

    aq_params = {"latitude": LAT, "longitude": LON, "hourly": "pm2_5", "timezone": "Asia/Kolkata", "start_date": start_date, "end_date": end_date}
    weather_params = {"latitude": LAT, "longitude": LON, "hourly": "temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m,cloud_cover", "timezone": "Asia/Kolkata", "start_date": start_date, "end_date": end_date}
    
    aq_response = requests.get(AQ_URL, params=aq_params)
    weather_response = requests.get(WEATHER_URL, params=weather_params)
    
    aq_data = aq_response.json()
    weather_data = weather_response.json()

    df_aq = pd.DataFrame(aq_data['hourly'])
    df_weather = pd.DataFrame(weather_data['hourly'])
    
    df = pd.merge(df_aq, df_weather, on='time')
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').interpolate(method='linear', limit_direction='forward')
    
    # Feature Engineering - MUST MATCH the notebook
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df.tail(INPUT_WINDOW)

# --- Main Application UI ---
st.title("🌬️ Noida PM2.5 Forecast (Next 72 Hours)")
st.markdown("Using a Transformer model to predict air quality based on the last 7 days of weather and pollution data.")

# Load model
model = load_keras_model()

# Button to trigger forecast
if st.button('Generate New Forecast', type="primary"):
    with st.spinner('Fetching latest data and running forecast... This can take a moment.'):
        
        # 1. Fetch data
        input_df = fetch_input_data()
        
        # 2. Scale data
        # IMPORTANT: The scaler must be fitted on the same feature order as in training
        feature_order = ['pm25', 'temperature_2m', 'relative_humidity_2m', 'precipitation', 'pressure_msl', 'wind_speed_10m', 'wind_direction_10m', 'cloud_cover', 'hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        input_df = input_df[feature_order]
        scaler = MinMaxScaler().fit(input_df) # Fit scaler on the fetched data
        data_scaled = scaler.transform(input_df)
        
        # 3. Predict
        input_tensor = np.expand_dims(data_scaled, axis=0)
        prediction_scaled = model.predict(input_tensor)
        
        # 4. Inverse transform
        target_col_idx = feature_order.index('pm25')
        dummy_array = np.zeros((OUTPUT_WINDOW, len(feature_order)))
        dummy_array[:, target_col_idx] = prediction_scaled[0, :]
        prediction_unscaled = scaler.inverse_transform(dummy_array)[:, target_col_idx]

        # 5. Display the results
        st.subheader("Forecast Results")
        forecast_times = pd.date_range(start=input_df.index[-1] + timedelta(hours=1), periods=OUTPUT_WINDOW, freq='H')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=input_df.index, y=input_df['pm25'], mode='lines', name='Historical PM2.5 (Input)', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=forecast_times, y=prediction_unscaled, mode='lines+markers', name='72-Hour Forecast', line=dict(color='red')))
        
        fig.update_layout(title="PM2.5 Forecast vs. Historical Data", xaxis_title="Date and Time", yaxis_title="PM2.5 (μg/m³)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("Forecast generated successfully!")
else:
    st.info("Click the button above to generate a 72-hour forecast.")
