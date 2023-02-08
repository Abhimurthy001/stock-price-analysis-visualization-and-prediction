import math
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
#import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')


st.title('Stock Forecast App')

stocks = ("BTC-USD","LBLOCK-USD", "RELIANCE.NS", "BHARTIARTL.NS", "ICICIBANK.NS", "TATASTEEL.NS", "ZEEL.NS")
selected_stock = st.selectbox("Select Stocks for prediction", stocks)



def load_data(ticker):
    data = yf.download(ticker)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
df_import = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Last Five Days')
st.write(df_import.tail())


def plot_raw_data():
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data.Date, y=data['Open'], name="stock_open",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(
        x=df_import.Date, y=df_import['Close'], name="stock_close", line_color='deepskyblue'))
    fig.layout.update(
        title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig


plot_raw_data()

# Model

# imports

data_load_state = st.text('Loading Model...')

data_main = df_import.filter(['Close'])
current_data = np.array(data_main).reshape(-1, 1).tolist()

n = 60
data_main.drop(data_main.tail(n).index, inplace = True)

df = np.array(data_main).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(np.array(df).reshape(-1, 1))
train_data = scaled_df[0: , :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

data_test = df_import.filter(['Close'])

df2 = np.array(data_test).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df2 = scaler.fit_transform(np.array(df2).reshape(-1, 1))

test_data = scaled_df2[-60:, :].tolist()
x_test = []
y_test = []
for i in range(60, 70):
    x_test = (test_data[i-60:i])
    x_test = np.asarray(x_test)
    pred_data = model.predict(x_test.reshape(1, x_test.shape[0], 1).tolist())

    y_test.append(pred_data[0][0])
    test_data.append(pred_data)


pred_next_10 = scaler.inverse_transform(np.asarray(y_test).reshape(-1, 1))

data_load_state.text('Loading Model... done!')


st.subheader("Next 10 Days")
st.write(pred_next_10)


# pred = current_data.extend(pred_next_10.tolist())


# plt.figure(figsize=(16, 8))
# plt.title('model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close_Price', fontsize=18)
# plt.plot(pred)
# plt.legend(['train'], loc='lower right')
# plt.show()
