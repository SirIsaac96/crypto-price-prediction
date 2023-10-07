# Import necessary libraries
import streamlit as st # for creating crypto price web app
import datetime as date
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from plotly import graph_objs as go 
import tensorflow as tf

# Create a start date
start = '2017-11-10'
today = date.datetime.today().strftime('%Y-%m-%d')

# Streamlit Web App
# Give the web app a title
st.title('Crypto Close Price Prediction Web App')

coins = ('BTC-USD', 'USDT-USD', 'LTC-USD', 'XRP-USD', 'ETH-USD')

# Create a select box to select between the different coins
selected_coin = st.selectbox('Select cryptocurrency dataset for prediction', coins)

# Create a function to load crypto data
def load_data(ticker):
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)  # put date in the first column
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
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'Crypto_Open'))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'Crypto_Close'))
    fig.layout.update(title_text = 'Crypto Price Data', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

# call the function
plot_raw_data()

# Forecasting Crypto Close Price using GRU
st.subheader('Forecast Crypto Close Price using GRU')

# Load the trained GRU model using load_model() function from Tensforflow Keras
gru_model = tf.keras.models.load_model('C:\\Trained models\\trained_gru_model.h5')

# Standardize the input data using MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
data_scaled = sc.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

# Function to create sequences for forecasting
def create_forecasting_sequences(data_scaled, target_col_index, n_steps, forecast_steps):
    x_seq, y_seq = [], []
    for i in range(len(data_scaled) - n_steps - forecast_steps, len(data_scaled) - n_steps):
        x_seq.append(data_scaled[i:i + n_steps, :])
        y_seq.append(data_scaled[i + n_steps, target_col_index])
    return np.array(x_seq), np.array(y_seq)

# Sequences for forecasting
n_tsteps = 1  # Number of time steps in each input sequence
features = 6  # Number of columns/features in each time step
target_col_index = 3 # 'Close' price is the target column
forecast_steps = 1 # forecast step for today
x_forecast, y_forecast = create_forecasting_sequences(data_scaled, target_col_index, n_tsteps, forecast_steps)

# Reshape the input data to fit the model's input shape
x_forecast = np.reshape(x_forecast, (x_forecast.shape[0], n_tsteps, features))

# Make predictions for today using the GRU model
today_prediction = gru_model.predict(x_forecast)[0, 0]

# Inverse transform the forecast prediction to get the actual price
today_price = sc.inverse_transform(np.hstack((data_scaled[-1, :target_col_index].reshape(1, -1),
                                              np.array([[today_prediction]]),
                                              data_scaled[-1, target_col_index + 1:].reshape(1, -1))))[:, target_col_index]

# Display forecast prediction for today
st.write("Today's Forecasted Close Price ($):")
st.write(today_price)
