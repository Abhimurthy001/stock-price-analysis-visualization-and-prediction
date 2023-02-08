import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2010-01-01"
TODAY = date.today().strftime("%y-%m-%d")

st.title("stock prediction app")

stocks=('AAPL','GOOG','MSFT','GME')
select_stocks=st.selectbox("select dataset for prediction",stocks)

n_years = st.slider("years of prediction:",1,4)
period=n_years*365

@st.cache
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("load data...")
data = load_data(select_stocks)
data_load_state.text("loading data...done !")

st.subheader('row data')
st.write(data.tail())

def plot_row_data():
    fig = go.figure()
    fig.add_trace(go.Scatter(x=data['Data'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Time series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_row_data()

df_train=data['Date','Close']
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2=m.plot_components(forecast)
st.write(fig2)
