import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Retail Sales Forecasting Dashboard")

uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    weekly = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    weekly.columns = ['ds', 'y']

    model = Prophet(yearly_seasonality=True)
    model.fit(weekly)

    future = model.make_future_dataframe(periods=20, freq='W')
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("Seasonality Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    st.subheader("Off-Season Uplift Simulation")

    uplift_percent = st.slider("Increase Off-Season Sales (%)", 0, 20, 5)

    forecast_sim = forecast.copy()
    forecast_sim['month'] = forecast_sim['ds'].dt.month

    mask = forecast_sim['month'].between(2, 9)
    forecast_sim.loc[mask, 'yhat'] *= (1 + uplift_percent/100)

    uplift_value = forecast_sim['yhat'].sum() - forecast['yhat'].sum()

    st.write("Projected Revenue Uplift:", round(uplift_value, 2))
