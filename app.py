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

# --- Helper Function for AQI Category & Color ---
def get_aqi_category(pm25):
    if pm25 <= 12.0: return ("Good", "#45E55A") # Green
    if pm25 <= 35.4: return ("Moderate", "#FCEA30") # Yellow
    if pm25 <= 55.4: return ("Unhealthy for Sensitive Groups", "#FF993D") # Orange
    if pm25 <= 150.4: return ("Unhealthy", "#FF4439") # Red
    if pm25 <= 250.4: return ("Very Unhealthy", "#A01BFF") # Purple
    return ("Hazardous", "#840026") # Maroon

# --- Caching Functions for Performance ---
@st.cache_resource
def load_keras_model():
    model = load_model('noida_pm25_transformer.keras')
    return model

@st.cache_data(ttl=3600)
def fetch_and_predict(model):
    # This function now contains the full logic and its result is cached
    tz = pytz.timezone("Asia/Kolkata")
    start_date = (datetime.now(tz) - timedelta(days=8)).strftime('%Y-%m-%d')
    end_date = datetime.now(tz).strftime('%Y-%m-%d')

    aq_params = {"latitude": LAT, "longitude": LON, "hourly": "pm2_5", "timezone": "Asia/Kolkata", "start_date": start_date, "end_date": end_date}
    weather_params = {"latitude": LAT, "longitude": LON, "hourly": "temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m,wind_direction_10m,cloud_cover", "timezone": "Asia/Kolkata", "start_date": start_date, "end_date": end_date}
    
    aq_data = requests.get(AQ_URL, params=aq_params).json()
    weather_data = requests.get(WEATHER_URL, params=weather_params).json()

    df_aq = pd.DataFrame(aq_data['hourly'])
    df_weather = pd.DataFrame(weather_data['hourly'])
    
    df = pd.merge(df_aq, df_weather, on='time')
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').interpolate(method='linear', limit_direction='forward')
    
    # Feature Engineering
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    input_df = df.tail(INPUT_WINDOW)
    
    # Scaling
    feature_order = ['pm25', 'temperature_2m', 'relative_humidity_2m', 'precipitation', 'pressure_msl', 'wind_speed_10m', 'wind_direction_10m', 'cloud_cover', 'hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    input_df_ordered = input_df[feature_order]
    scaler = MinMaxScaler().fit(input_df_ordered)
    data_scaled = scaler.transform(input_df_ordered)
    
    # Prediction
    input_tensor = np.expand_dims(data_scaled, axis=0)
    prediction_scaled = model.predict(input_tensor)
    
    # Inverse transform
    target_col_idx = feature_order.index('pm25')
    dummy_array = np.zeros((OUTPUT_WINDOW, len(feature_order)))
    dummy_array[:, target_col_idx] = prediction_scaled[0, :]
    prediction_unscaled = scaler.inverse_transform(dummy_array)[:, target_col_idx]
    
    forecast_times = pd.date_range(start=input_df.index[-1] + timedelta(hours=1), periods=OUTPUT_WINDOW, freq='H')
    forecast_df = pd.DataFrame({'Time': forecast_times, 'Predicted PM2.5': prediction_unscaled})
    
    return input_df, forecast_df

# --- Main Application UI ---
st.title("🌬️ Real-Time Noida Air Quality Forecast")
st.markdown(f"Automatic 72-hour forecast using a Transformer model. Last updated: **{datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %I:%M %p')}**")

# Load model and run forecast automatically
model = load_keras_model()
with st.spinner('Generating latest 72-hour forecast...'):
    historical_df, forecast_df = fetch_and_predict(model)

st.success("Forecast generated successfully!")

# --- Display AQI Cards ---
st.subheader("Hourly AQI Card View")
cols = st.columns(6) # Create a grid with 6 columns

for i, row in forecast_df.iterrows():
    hour = row['Time']
    pm25 = row['Predicted PM2.5']
    category, color = get_aqi_category(pm25)
    
    with cols[i % 6]:
        st.markdown(
            f"""
            <div style="background-color: {color}; border-radius: 10px; padding: 15px; text-align: center; color: white; margin-bottom: 10px;">
                <h3 style="margin:0; font-size: 1.2em;">{hour.strftime('%b %d, %I%p')}</h3>
                <h2 style="margin:0; font-size: 2em;">{pm25:.1f}</h2>
                <p style="margin:0;">{category}</p>
            </div>
            """, unsafe_allow_html=True)

# --- Display Graph ---
st.subheader("Forecast Graph")

# Create a list of colors for the markers based on the prediction
marker_colors = [get_aqi_category(pm25)[1] for pm25 in forecast_df['Predicted PM2.5']]

fig = go.Figure()
# Add historical trace
fig.add_trace(go.Scatter(x=historical_df.index, y=historical_df['pm25'], mode='lines', name='Historical PM2.5 (Last 7 Days)', line=dict(color='gray')))
# Add forecast trace with colored markers
fig.add_trace(go.Scatter(x=forecast_df['Time'], y=forecast_df['Predicted PM2.5'], mode='lines+markers', name='72-Hour Forecast', 
                         line=dict(color='royalblue', width=3),
                         marker=dict(color=marker_colors, size=8, symbol='circle')))

fig.update_layout(
    title="PM2.5 Forecast vs. Historical Data",
    xaxis_title="Date and Time",
    yaxis_title="PM2.5 Concentration (μg/m³)",
    legend_title="Legend"
)
st.plotly_chart(fig, use_container_width=True)
