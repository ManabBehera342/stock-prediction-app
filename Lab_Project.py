import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st

# Set Alpha Vantage API Key
API_KEY = "N3O1A96PBPG1UPDH"

# List of stock symbols for dropdown menu (Expanded to ~100 stocks)
STOCK_OPTIONS = [
    "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NFLX", "TCS.BSE", "RELIANCE.BSE", "INFY.BSE", 
    "HDFC.BSE", "ICICIBANK.BSE", "SBIN.BSE", "TATASTEEL.BSE", "HINDUNILVR.BSE", "ITC.BSE", "BAJFINANCE.BSE", "ADANIPORTS.BSE",
    "CIPLA.BSE", "DRREDDY.BSE", "KOTAKBANK.BSE", "LT.BSE", "MARUTI.BSE", "NESTLEIND.BSE", "ONGC.BSE", "POWERGRID.BSE", 
    "SUNPHARMA.BSE", "TATAMOTORS.BSE", "ULTRACEMCO.BSE", "WIPRO.BSE", "ZEEL.BSE", "BHARTIARTL.BSE", "AXISBANK.BSE", "COALINDIA.BSE",
    "DABUR.BSE", "EICHERMOT.BSE", "GAIL.BSE", "GRASIM.BSE", "HCLTECH.BSE", "HDFCBANK.BSE", "HEROMOTOCO.BSE", "HINDALCO.BSE", 
    "INDUSINDBK.BSE", "JSWSTEEL.BSE", "M&M.BSE", "NTPC.BSE", "TITAN.BSE", "UPL.BSE", "VEDL.BSE", "YESBANK.BSE", "BHEL.BSE",
    "BPCL.BSE", "GODREJCP.BSE", "HAVELLS.BSE", "ICICIGI.BSE", "LUPIN.BSE", "MINDTREE.BSE", "NMDC.BSE", "PETRONET.BSE", "RBLBANK.BSE",
    "SIEMENS.BSE", "SRF.BSE", "TORNTPHARM.BSE", "TATACHEM.BSE", "BIOCON.BSE", "HAL.BSE", "IRCTC.BSE", "MPHASIS.BSE", "PIIND.BSE",
    "RECLTD.BSE", "TATACONSUM.BSE", "UCOBANK.BSE", "VOLTAS.BSE", "ZOMATO.BSE", "POLYCAB.BSE", "PERSISTENT.BSE", "DMART.BSE",
    "ICICIPRULI.BSE", "HDFCAMC.BSE", "BANKBARODA.BSE", "NAVINFLUOR.BSE", "ASTRAL.BSE", "INDIGO.BSE", "CROMPTON.BSE", "IDFCFIRSTB.BSE"
]

def get_stock_data(symbol):
    try:
        ts = TimeSeries(key=API_KEY, output_format="pandas")
        data, _ = ts.get_intraday(symbol=symbol, interval="5min", outputsize="full")
        data.columns = ["Open", "High", "Low", "Close", "Volume"]
        return data[::-1]  # Reverse order
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[["Close"]])
    return data, scaler

def predict_future_price(model, data, scaler):
    last_60_days = data["Close"].values[-60:].reshape(-1, 1)  # Take last 60 data points
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = np.array([last_60_days_scaled]).reshape(1, 60, 1)
    predicted_price_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    return predicted_price[0][0]

def get_investment_advice(current_price, predicted_price):
    price_diff = predicted_price - current_price
    percentage_change = (price_diff / current_price) * 100
    if percentage_change > 2:
        return "BUY ✅", percentage_change
    elif percentage_change < -2:
        return "SELL ❌", percentage_change
    else:
        return "HOLD ⚖️", percentage_change

def calculate_moving_average(data, window):
    return data["Close"].rolling(window=window).mean()

def calculate_volatility(data, window=60):
    return data["Close"].rolling(window=window).std()

# Streamlit Web App
st.title("📈 Real-Time Stock Market Prediction")

# Dropdown menu for stock selection
stock_symbol = st.selectbox("Select Stock Symbol:", STOCK_OPTIONS)

if st.button("Predict Stock Price"):
    st.write(f"Fetching real-time data for {stock_symbol}...")
    stock_data = get_stock_data(stock_symbol)
    data, scaler = preprocess_data(stock_data)
    
    model = load_model("stock_price_lstm_model.h5")  # Load saved model
    
    predicted_price = predict_future_price(model, stock_data, scaler)
    current_price = stock_data["Close"].iloc[-1]
    
    advice, percentage_change = get_investment_advice(current_price, predicted_price)
    
    st.write(f"📌 Current Price: **${current_price:.2f}**")
    st.write(f"📊 Predicted Price (Next Day): **${predicted_price:.2f}**")
    st.write(f"📈 Expected Change: **{percentage_change:.2f}%**")
    st.write(f"💡 Investment Advice: **{advice}**")
    
    st.line_chart(stock_data["Close"])
    
