import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Netflix Stock Forecast", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = joblib.load("netflix_stock.joblib")
    df.index = pd.to_datetime(df.index)
    return df

df = load_data()

# Sidebar for user inputs
st.sidebar.header("Forecast Parameters")
forecast_days = st.sidebar.slider("Select number of days to forecast:", 30, 365, 90)
test_size = st.sidebar.slider("Select test set size (days):", 30, 180, 60)
model_choice = st.sidebar.selectbox("Select Model:", ["ARIMA", "Simple Exponential Smoothing"])

# Main app
st.title("Netflix Stock Price Analysis & Forecasting")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(df)

# Plot closing price
st.subheader("Netflix Closing Price Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df['Close'], color='purple', ax=ax)
plt.title('Netflix Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
st.pyplot(fig)

# Time series decomposition
st.subheader("Time Series Decomposition")
decomposition = seasonal_decompose(df['Close'], model='additive', period=30)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
decomposition.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')
decomposition.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')
decomposition.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal')
decomposition.resid.plot(ax=ax4)
ax4.set_ylabel('Residual')
plt.tight_layout()
st.pyplot(fig)

# Stationarity check
st.subheader("Stationarity Check")
result = adfuller(df['Close'])
st.write(f'ADF Statistic: {result[0]:.4f}')
st.write(f'p-value: {result[1]:.4f}')
st.write('Critical Values:')
for key, value in result[4].items():
    st.write(f'   {key}: {value:.4f}')

if result[1] > 0.05:
    st.warning("The time series is not stationary (p-value > 0.05)")
else:
    st.success("The time series is stationary (p-value ≤ 0.05)")

# Forecasting section
st.subheader("Forecasting")

# Prepare data
train_size = len(df) - test_size
train, test = df['Close'][:train_size], df['Close'][train_size:]

if model_choice == "ARIMA":
    st.write("### ARIMA Model")
    
    # Fit ARIMA model
    try:
        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()
        
        # Forecast
        forecast = model_fit.forecast(steps=forecast_days)
        forecast_index = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_days)
        forecast_series = pd.Series(forecast, index=forecast_index)
        
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(train.index, train, label='Training')
        plt.plot(test.index, test, label='Actual')
        plt.plot(forecast_series.index, forecast_series, label='Forecast', color='red')
        plt.title('Netflix Stock Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        st.pyplot(fig)
        
        # Calculate RMSE
        predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
        rmse = sqrt(mean_squared_error(test, predictions))
        st.write(f"RMSE: {rmse:.2f}")
        
    except Exception as e:
        st.error(f"Error in ARIMA model: {str(e)}")

elif model_choice == "Simple Exponential Smoothing":
    st.write("### Simple Exponential Smoothing")
    
    try:
        # Simple implementation of exponential smoothing
        alpha = 0.2
        predictions = []
        history = [x for x in train]
        
        for t in range(len(test)):
            yhat = alpha * history[-1] + (1 - alpha) * history[-1]
            predictions.append(yhat)
            history.append(test[t])
        
        # Forecast future values
        forecast = []
        last_value = df['Close'].iloc[-1]
        
        for _ in range(forecast_days):
            next_val = alpha * last_value + (1 - alpha) * last_value
            forecast.append(next_val)
            last_value = next_val
            
        forecast_index = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_days)
        forecast_series = pd.Series(forecast, index=forecast_index)
        
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(train.index, train, label='Training')
        plt.plot(test.index, test, label='Actual')
        plt.plot(test.index, predictions, label='Predictions', color='green')
        plt.plot(forecast_series.index, forecast_series, label='Forecast', color='red')
        plt.title('Netflix Stock Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        st.pyplot(fig)
        
        # Calculate RMSE
        rmse = sqrt(mean_squared_error(test, predictions))
        st.write(f"RMSE: {rmse:.2f}")
        
    except Exception as e:
        st.error(f"Error in Simple Exponential Smoothing model: {str(e)}")

# Show forecast values
if st.checkbox("Show Forecast Values"):
    st.subheader("Forecasted Values")
    try:
        forecast_df = pd.DataFrame({
            'Date': forecast_series.index,
            'Forecasted Price': forecast_series.values
        })
        st.write(forecast_df)
    except:
        st.warning("No forecast available yet. Please run a forecast model first.")

# Data statistics
st.subheader("Data Statistics")
st.write(df.describe())

# Notes
st.subheader("Notes")
st.write("""
- This app provides basic time series analysis and forecasting for Netflix stock prices.
- The ARIMA model may take some time to fit, especially with larger datasets.
- For better results, consider more advanced models like Prophet or LSTM networks.
- Always validate forecasts with additional analysis before making investment decisions.
""")