import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

# --- Page Configuration ---
st.set_page_config(
    page_title="Noida Air Quality Forecast",
    page_icon="https://raw.githubusercontent.com/RevMax-creator/AQI_Transformer/main/favicon.ico", 
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
    if pm25 <= 12.0: return ("Good", "#45E55A")
    if pm25 <= 35.4: return ("Moderate", "#FCEA30")
    if pm25 <= 55.4: return ("Unhealthy for Sensitive Groups", "#FF993D")
    if pm25 <= 150.4: return ("Unhealthy", "#FF4439")
    if pm25 <= 250.4: return ("Very Unhealthy", "#A01BFF")
    return ("Hazardous", "#840026")

# --- Caching Functions for Performance ---
@st.cache_resource
def load_keras_model():
    model = load_model('noida_pm25_transformer.keras', compile=False)
    return model

@st.cache_data(ttl=3600)
def fetch_and_predict(_model): # Added underscore to bypass Streamlit hashing for keras model
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)
    
    # --- Adjust end_date if before 9 AM IST ---
    if now.hour < 9:
        end_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')
        st.info(f"Current time is before 9 AM IST. Using data up to {end_date} for complete daily coverage.")
    else:
        end_date = now.strftime('%Y-%m-%d')
    
    # Fetch 8 days back to give enough buffer to calculate the 24h rolling windows
    start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=8)).strftime('%Y-%m-%d')
    
    # --- API Params ---
    aq_params = {
        "latitude": LAT, "longitude": LON, 
        "hourly": "pm10,pm2_5,nitrogen_dioxide,ozone",
        "timezone": "Asia/Kolkata", "start_date": start_date, "end_date": end_date
    }
    
    weather_params = {
        "latitude": LAT, "longitude": LON, 
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,rain,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,cloud_cover",
        "timezone": "Asia/Kolkata", "start_date": start_date, "end_date": end_date
    }
    
    # --- Error Handling ---
    aq_response = requests.get(AQ_URL, params=aq_params)
    if not aq_response.ok:
        st.error(f"Failed to fetch Air Quality data. API returned: {aq_response.text}")
        st.stop()

    weather_response = requests.get(WEATHER_URL, params=weather_params)
    if not weather_response.ok:
        st.error(f"Failed to fetch Weather data. API returned: {weather_response.text}")
        st.stop()

    aq_data = aq_response.json()
    weather_data = weather_response.json()
    
    if 'hourly' not in weather_data or 'hourly' not in aq_data:
        st.error("The data returned from the API was not in the expected format.")
        st.stop()

    df_aq = pd.DataFrame(aq_data['hourly'])
    df_aq = df_aq.rename(columns={'pm2_5': 'pm25', 'pm10': 'pm10', 'nitrogen_dioxide': 'no2', 'ozone': 'o3'})
    
    df_weather = pd.DataFrame(weather_data['hourly'])
    df_weather = df_weather.rename(columns={
        'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity',
        'dew_point_2m': 'dew_point', 'wind_speed_10m': 'wind_speed',
        'wind_direction_10m': 'wind_direction', 'wind_gusts_10m': 'wind_gusts'
    })

    df = pd.merge(df_aq, df_weather, on='time')
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').interpolate(method='linear', limit_direction='forward')
    
    # ---------------------------------------------------------
    # FIX 1: Calculate Rolling Features BEFORE slicing the window
    # ---------------------------------------------------------
    df['pm25_rolling_mean_24h'] = df['pm25'].rolling(window=24).mean()
    df['pm25_rolling_std_24h'] = df['pm25'].rolling(window=24).std()
    df['wind_speed_drop_6h'] = df['wind_speed'].diff(periods=6)
    df['temp_drop_6h'] = df['temperature'].diff(periods=6)
    
    # Backfill the first 24 hours of NaNs to avoid dropping data
    df = df.bfill().ffill()

    # Time Features
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Now extract exactly the last 7 days (168 hours) for the model
    input_df = df.tail(INPUT_WINDOW).copy()
    historical_pm25_raw = input_df['pm25'].copy()
    
    # Define exact feature splits
    continuous_cols = [
        'pm25', 'pm10', 'no2', 'o3', 'temperature', 'humidity',
        'dew_point', 'precipitation', 'rain', 'pressure_msl',
        'surface_pressure', 'wind_speed', 'wind_direction', 'wind_gusts', 'cloud_cover',
        'pm25_rolling_mean_24h', 'pm25_rolling_std_24h', 'wind_speed_drop_6h', 'temp_drop_6h'
    ]
    
    cyclical_cols = [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ]
    
    feature_order = continuous_cols + cyclical_cols
    
    # ---------------------------------------------------------
    # FIX 2: Apply scaler ONLY to continuous columns
    # ---------------------------------------------------------
    scaler = joblib.load('transformer_scaler.pkl')
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])
    
    # Ensure correct column ordering for the neural network
    final_input_data = input_df[feature_order].values
    input_tensor = np.expand_dims(final_input_data, axis=0)
    
    input_df['pm25'] = historical_pm25_raw
    # Predict
    prediction_scaled = _model.predict(input_tensor)
    
    # ---------------------------------------------------------
    # FIX 3: Dummy array must match the 19 continuous columns
    # ---------------------------------------------------------
    target_scaler_idx = continuous_cols.index('pm25') # This will be index 0
    dummy_array = np.zeros((OUTPUT_WINDOW, len(continuous_cols))) 
    
    dummy_array[:, target_scaler_idx] = prediction_scaled[0, :]
    prediction_unscaled = scaler.inverse_transform(dummy_array)[:, target_scaler_idx]
    
    forecast_times = pd.date_range(start=input_df.index[-1] + timedelta(hours=1), periods=OUTPUT_WINDOW, freq='h')
    forecast_df = pd.DataFrame({'Time': forecast_times, 'Predicted PM2.5': prediction_unscaled})
    
    return input_df, forecast_df

# --- Main Application UI ---
st.title("🌬️ Real-Time Noida Air Quality Forecast")
st.markdown(f"Automatic 72-hour forecast using a Transformer model. Last updated: **{datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %I:%M %p')}**")

model = load_keras_model()
with st.spinner('Generating latest 72-hour forecast...'):
    historical_df, forecast_df = fetch_and_predict(model)

st.success("Forecast generated successfully!")

st.subheader("Hourly AQI Card View")
cols = st.columns(6)

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

st.subheader("Forecast Graph")
marker_colors = [get_aqi_category(pm25)[1] for pm25 in forecast_df['Predicted PM2.5']]
fig = go.Figure()
fig.add_trace(go.Scatter(x=historical_df.index, y=historical_df['pm25'], mode='lines', name='Historical PM2.5 (Last 7 Days)', line=dict(color='gray')))
fig.add_trace(go.Scatter(x=forecast_df['Time'], y=forecast_df['Predicted PM2.5'], mode='lines+markers', name='72-Hour Forecast', 
                         line=dict(color='royalblue', width=3),
                         marker=dict(color=marker_colors, size=8, symbol='circle')))
fig.update_layout(title="PM2.5 Forecast vs. Historical Data", xaxis_title="Date and Time", yaxis_title="PM2.5 (μg/m³)", legend_title="Legend")
st.plotly_chart(fig, use_container_width=True)