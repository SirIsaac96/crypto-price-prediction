# Import necessary libraries
import streamlit as st # for creating crypto price web app
import datetime as date
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly import graph_objs as go 
import tensorflow as tf
import pandas as pd

# Create a start date
start = '2017-11-10'
today = date.datetime.today().strftime('%Y-%m-%d')

# Streamlit Web App
# Give the web app a title
st.title('Crypto Close Price Prediction')

# Define the cryptocurrencies
coins = ('BTC-USD', 'USDT-USD', 'LTC-USD', 'XRP-USD', 'ETH-USD')

# Create a select box to select between the different coins
selected_coin = st.selectbox('Select cryptocurrency dataset for prediction', coins)

# Create a function to load crypto data
def load_data(ticker):
    data = yf.download(ticker, start, today)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data

# Call the function
data_load_state = st.text('Load data...')
data = load_data(selected_coin)

# reset the text
data_load_state.text('Loading data...done!')

# Plot raw the data
# raw pandas dataframe
st.subheader('Raw data')
st.write(data.tail())  # tail of the dataframe  

# plot the raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'Crypto_Open', line = dict(color = 'red')))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'Crypto_Close', line = dict(color = 'blue')))
    fig.layout.update(title_text = 'Crypto Market Behavior Over Time', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

# call the function
plot_raw_data()

# Forecasting Crypto Close Price using GRU Model
st.subheader('Forecasting Crypto Close Price using GRU Model')

# Load the trained GRU models using load_model() function from Tensforflow Keras
# gru_model_daily = tf.keras.models.load_model('C:/Trained models/trained_gru_model.h5')
gru_model_daily = tf.keras.models.load_model('trained_gru_model_daily.keras')
gru_model_weekly = tf.keras.models.load_model('trained_gru_model_weekly.keras')
gru_model_monthly = tf.keras.models.load_model('trained_gru_model_monthly.keras')

# Standardize the input data using MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

# Copy 'Adj Close' from 'Close' if it does not exist
if 'Adj Close' not in data.columns:
    data['Adj Close'] = data['Close']

# Define and scale using all 6 expected columns
required_columns = ['Close', 'High', 'Low', 'Open', 'Adj Close', 'Volume']
data_scaled = sc.fit_transform(data[required_columns])
features = len(required_columns)
target_col_index = required_columns.index('Close')

# Function to create sequences for forecasting
def create_forecasting_sequences(data_scaled, target_col_index, n_tsteps, forecast_steps):
    x_seq, y_seq = [], []
    for i in range(len(data_scaled) - n_tsteps - forecast_steps, len(data_scaled) - n_tsteps):
        x_seq.append(data_scaled[i:i + n_tsteps, :])
        y_seq.append(data_scaled[i + n_tsteps, target_col_index])
    return np.array(x_seq), np.array(y_seq)

features = 6  # Number of columns/features in each time step
target_col_index = 0 # 'Close' price is the target column

# Sequences for daily forecasting
n_tsteps_daily = 1  # Number of time steps in each input sequence
forecast_steps_daily = 1 # daily forecast steps
x_forecast_daily, y_forecast_daily = create_forecasting_sequences(data_scaled, target_col_index, 
                                                                  n_tsteps_daily, forecast_steps_daily)

# Reshape the input data to fit the model's input shape
x_forecast_daily = np.reshape(x_forecast_daily, (x_forecast_daily.shape[0], n_tsteps_daily, features))

# Make daily predictions using the GRU model
daily_prediction = gru_model_daily.predict(x_forecast_daily)[0, 0]

# Inverse transform the predicted price to get the actual price
daily_price = sc.inverse_transform(np.hstack((data_scaled[-1, :target_col_index].reshape(1, -1),
                                              np.array([[daily_prediction]]),
                                              data_scaled[-1, target_col_index + 1:].reshape(1, -1))))[:, target_col_index]

# Display daily forecasted close price in $
st.write("Daily Forecasted Close Price ($):")
st.write(daily_price)

# Sequences for weekly forecasting
n_tsteps_weekly = 7  # Number of time steps in each input sequence
forecast_steps_weekly = 7  # weekly forecast steps
x_forecast_weekly, y_forecast_weekly = create_forecasting_sequences(data_scaled, target_col_index, 
                                                                    n_tsteps_weekly, forecast_steps_weekly)

# Reshape the input data to fit the model's input shape
x_forecast_weekly = np.reshape(x_forecast_weekly, (x_forecast_weekly.shape[0], n_tsteps_weekly, features))

# Make weekly predictions using the GRU model
weekly_prediction = gru_model_weekly.predict(x_forecast_weekly)[0, 0]

# Inverse transform the forecast predictions to get the actual prices
weekly_price = sc.inverse_transform(np.hstack((data_scaled[-7, :target_col_index].reshape(1, -1),
                                                np.array([[weekly_prediction]]),
                                                data_scaled[-7, target_col_index + 1:].reshape(1, -1))))[:, target_col_index]

# Display weekly forecasted close price in $
st.write("Weekly Forecasted Close Price ($):")
st.write(weekly_price)

# Sequences for monthly forecasting
n_tsteps_monthly = 30  # Number of time steps in each input sequence
forecast_steps_monthly = 30  # monthly forecast steps
x_forecast_monthly, y_forecast_monthly = create_forecasting_sequences(data_scaled, target_col_index, 
                                                                    n_tsteps_monthly, forecast_steps_monthly)

# Reshape the input data to fit the model's input shape
x_forecast_monthly = np.reshape(x_forecast_monthly, (x_forecast_monthly.shape[0], n_tsteps_monthly, features))

# Make monthly predictions using the GRU model
monthly_prediction = gru_model_monthly.predict(x_forecast_monthly)[0, 0]

# Inverse transform the forecast predictions to get the actual prices
monthly_price = sc.inverse_transform(np.hstack((data_scaled[-30, :target_col_index].reshape(1, -1),
                                                np.array([[monthly_prediction]]),
                                                data_scaled[-30, target_col_index + 1:].reshape(1, -1))))[:, target_col_index]

# Display monthly forecasted close price in $
st.write("Monthly Forecasted Close Price ($):")
st.write(monthly_price)
