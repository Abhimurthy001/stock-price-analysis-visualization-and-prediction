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
