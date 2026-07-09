import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys

# 1. Turn OFF CPU-specific oneDNN flags so it doesn't conflict with GPU memory allocations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
import logging
# Set log level to suppress WARNINGs (0 = all, 1 = no INFO, 2 = no WARNING, 3 = no ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# 2. Force TensorFlow to visible recognize and register your GTX 1660 Ti GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable dynamic memory growth so TensorFlow doesn't freeze your entire desktop display
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        st.sidebar.success(f"🚀 GPU Active: {len(gpus)} NVIDIA Card(s) Found!")
    except RuntimeError as e:
        st.sidebar.error(f"GPU Error: {e}")
else:
    st.sidebar.warning("⚠️ Running on CPU mode. NVIDIA CUDA drivers not linked.")

# Suppress python-level warnings
tf.get_logger().setLevel(logging.ERROR)
tf.keras.utils.set_random_seed(1)
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
import datetime
from datetime import date
from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import GRU, LSTM, Dense, Dropout, AdditiveAttention, Flatten
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model

# Defining a Sidebar
st.sidebar.title("Stock Price Prediction :")
st.sidebar.write('Copyright by BB_Chan')
code = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL,0005.hk,...) :")
start = st.sidebar.date_input("Select Start Date",value=datetime.date(2020,1,1),
                      min_value=datetime.date(1990,1,1),
                      max_value=datetime.date(2026,12,31))
end = st.sidebar.date_input("Select End Date",value=datetime.date(2026,12,31))
today = date.today()
if end > today:
    end = today

signal_days = st.sidebar.number_input("Select Last Trading Signal Days :",1,10,5,1)
pred_days = st.sidebar.number_input("Select Prediction Days :",1,20,10,1)
# Defining Checkbox
st.sidebar.write('Select Technical Indicator(s) :')
MACD_DMI = st.sidebar.checkbox('Trend : MACD & DMI')
RSI_KDJ = st.sidebar.checkbox('Momentum : RSI & KDJ')
BB_BIAS = st.sidebar.checkbox('Volatility : BB & BIAS')
# Defining Checkbox
st.sidebar.write('Select Prediction Model(s) :')
New_XGB = st.sidebar.checkbox('Create new XGB')
Rel_XGB = st.sidebar.checkbox('Load saved XGB')
New_GRU = st.sidebar.checkbox('Create new GRU')
Rel_GRU = st.sidebar.checkbox('Load saved GRU')
New_LSTM = st.sidebar.checkbox('Create new LSTM')
Rel_LSTM = st.sidebar.checkbox('Load saved LSTM')
New_LSTM_AM = st.sidebar.checkbox('Create new LSTM - Attention')
Rel_LSTM_AM = st.sidebar.checkbox('Load saved LSTM - Attention')
New_LSTM_FEAT = st.sidebar.checkbox('Create new LSTM - Features')
Rel_LSTM_FEAT = st.sidebar.checkbox('Load saved LSTM - Features')
# Defining a Button
button = st.sidebar.button('Submit')
if not button:
    st.stop()

stock = yf.download(code, start, end).droplevel('Ticker',axis=1)
stock.to_csv(code + '.csv')
df = pd.read_csv(code + '.csv')
st.header(code)
st.subheader('Stock Data')

st.dataframe(df)
start_time = datetime.datetime.now()

# Calculate Moving Averages
df['SMA10'] = df['Close'].rolling(window=10).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['EMA10'] = df['Close'].ewm(span=10).mean()
df['EMA50'] = df['Close'].ewm(span=50).mean()
df['EMA100'] = df['Close'].ewm(span=100).mean()

# Calculate Moving Average Convergence Divergence
df['EMA1'] = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
df['EMA2'] = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
df['DIF'] = df['EMA1'] - df['EMA2']
df['DEA'] = df['DIF'].ewm(span=9, adjust=False, min_periods=9).mean()
df['MACD'] = 2 * (df['DIF'] - df['DEA'])

# Calculate Bollinger BANDS
def BBANDS(df0, ma_days=20, std_dev = 2):
    ma = df0.Close.rolling(window=ma_days).mean()
    sd = df0.Close.rolling(window=ma_days).std()
    df0['MiddleBand'] = ma
    df0['UpperBand'] = ma + (std_dev * sd)
    df0['LowerBand'] = ma - (std_dev * sd)
    return df0
df = BBANDS(df)

# Calculate Directional Movement Index
def get_adx(high, low, close, lookback=14):
    """
            Vectorized Directional Movement Index (ADX/DMI).
    """
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Efficient True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(lookback).mean()
    # Smooth DM using Wilder's method (alpha=1/N)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / lookback, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / lookback, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1 / lookback, adjust=False).mean()
    return plus_di, minus_di, adx
df['Plus_di'], df['Minus_di'], df['ADX'] = get_adx(df['High'], df['Low'], df['Close'], 14)

# Calculate KDJ
def calKDJ(df2,n=9,m1=3,m2=3):
    """
        Vectorized KDJ calculation (Stochastic Oscillator).
        Default parameters: 9-day window, 3-day smoothing.
    """
    # Calculate the Rolling Low and High
    low_list = df2['Low'].rolling(window=n, min_periods=n).min()
    high_list = df2['High'].rolling(window=n, min_periods=n).max()
    # Calculate RSV (Raw Stochastic Value)
    rsv = (df2['Close'] - low_list) / (high_list - low_list) * 100
    # Calculate K and D using EWM (com=period-1 is equivalent to the 1/N smoothing)
    # formula: K = 2/3 * K_prev + 1/3 * RSV
    df2['K'] = rsv.ewm(com=m1 - 1, adjust=False).mean()
    df2['D'] = df2['K'].ewm(com=m2 - 1, adjust=False).mean()
    # Calculate J
    df2['J'] = 3 * df2['K'] - 2 * df2['D']
    return df2
df = calKDJ(df)

# Calculate Relative Strength Index
def calRSI(df3, periodList):
    """
        Vectorized Relative Strength Index calculation for multiple periods.
    """
    # 1. Calculate price changes once
    delta = df3['Close'].diff()
    # 2. Separate gains and losses once
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    # 3. Loop through each period in the list
    for period in periodList:
        # Calculate EMA of gains and losses for this specific period
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        # Create a specific column for each period
        df3[f'RSI{period}'] = 100 - (100 / (1 + rs))
    return df3
periods = [6, 12, 24]
df = calRSI(df, periods)

# Calculate BIAS
def calBIAS(df1, periodList):
    # Review periods: 6, 12 & 24days
    for period in periodList:
        df1['MA' + str(period)] = df1['Close'].rolling(window=period).mean()
        df1['MA' + str(period)].fillna(value=df['Close'], inplace=True)
        df1['BIAS' + str(period)] = (df1['Close'] - df1['MA' + str(period)]) / df1['MA' + str(period)] * 100
    return df1
df = calBIAS(df, [6, 12, 24])

# Calculate Support & Resistance
WINDOW = 30
df['min'] = round((df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=WINDOW)[0]]['Close']),2)
df['max'] = round((df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=WINDOW)[0]]['Close']),2)
# ###  Calculate Support Levels:
SupportDate = ''
for cnt in range(len(df)-1,max(len(df)-90,29),-1):
    if df.iloc[cnt]['min'] > 0:
        SupportDate += f"{df.iloc[cnt]['min']} ({df.iloc[cnt]['Date']}),"
# ###  Calculate Resistance Levels:
ResistDate = ''
for cnt in range(len(df)-1,max(len(df)-60,29),-1):
    if df.iloc[cnt]['max'] > 0:
        ResistDate += f"{df.iloc[cnt]['max']} ({df.iloc[cnt]['Date']}), "

# ###################################
# Rule-Based Trading Signals Processing Loops
EMA10buyDate, EMA10sellDate = "", ""
EMA50buyDate, EMA50sellDate = "", ""
EMA100buyDate, EMA100sellDate = "", ""
BBbuyDate, BBsellDate = "", ""
MACDbuyDate, MACDsellDate = "", ""
DMIbuyDate, DMIsellDate = "", ""
BIASbuyDate, BIASsellDate = "", ""
KDJbuyDate, KDJsellDate = "", ""
RSIbuyDate, RSIsellDate = "", ""

start_idx = len(df) - signal_days - 2
end_idx = len(df) - 2

for cnt in range(start_idx, end_idx):
    if cnt < 30: continue

    # EMA 10
    if (df.iloc[cnt]['Close'] < df.iloc[cnt + 1]['Close'] < df.iloc[cnt + 2]['Close']) and \
            (df.iloc[cnt]['EMA10'] < df.iloc[cnt + 1]['EMA10'] < df.iloc[cnt + 2]['EMA10']) and \
            (df.iloc[cnt + 1]['EMA10'] > df.iloc[cnt]['Close']) and (
            df.iloc[cnt + 2]['EMA10'] < df.iloc[cnt + 1]['Close']):
        EMA10buyDate += df.iloc[cnt]['Date'] + ', '
    if (df.iloc[cnt]['Close'] > df.iloc[cnt + 1]['Close'] > df.iloc[cnt + 2]['Close']) and \
            (df.iloc[cnt]['EMA10'] > df.iloc[cnt + 1]['EMA10'] > df.iloc[cnt + 2]['EMA10']) and \
            (df.iloc[cnt + 1]['EMA10'] < df.iloc[cnt]['Close']) and (
            df.iloc[cnt + 2]['EMA10'] > df.iloc[cnt + 1]['Close']):
        EMA10sellDate += df.iloc[cnt]['Date'] + ', '

    # EMA 50
    if (df.iloc[cnt]['Close'] < df.iloc[cnt + 1]['Close'] < df.iloc[cnt + 2]['Close']) and \
            (df.iloc[cnt]['EMA50'] < df.iloc[cnt + 1]['EMA50'] < df.iloc[cnt + 2]['EMA50']) and \
            (df.iloc[cnt + 1]['EMA50'] > df.iloc[cnt]['Close']) and (
            df.iloc[cnt + 2]['EMA50'] < df.iloc[cnt + 1]['Close']):
        EMA50buyDate += df.iloc[cnt]['Date'] + ', '
    if (df.iloc[cnt]['Close'] > df.iloc[cnt + 1]['Close'] > df.iloc[cnt + 2]['Close']) and \
            (df.iloc[cnt]['EMA50'] > df.iloc[cnt + 1]['EMA50'] > df.iloc[cnt + 2]['EMA50']) and \
            (df.iloc[cnt + 1]['EMA50'] < df.iloc[cnt]['Close']) and (
            df.iloc[cnt + 2]['EMA50'] > df.iloc[cnt + 1]['Close']):
        EMA50sellDate += df.iloc[cnt]['Date'] + ', '

    # EMA 100
    if (df.iloc[cnt]['Close'] < df.iloc[cnt + 1]['Close'] < df.iloc[cnt + 2]['Close']) and \
            (df.iloc[cnt]['EMA100'] < df.iloc[cnt + 1]['EMA100'] < df.iloc[cnt + 2]['EMA100']) and \
            (df.iloc[cnt + 1]['EMA100'] > df.iloc[cnt]['Close']) and (
            df.iloc[cnt + 2]['EMA100'] < df.iloc[cnt + 1]['Close']):
        EMA100buyDate += df.iloc[cnt]['Date'] + ', '
    if (df.iloc[cnt]['Close'] > df.iloc[cnt + 1]['Close'] > df.iloc[cnt + 2]['Close']) and \
            (df.iloc[cnt]['EMA100'] > df.iloc[cnt + 1]['EMA100'] > df.iloc[cnt + 2]['EMA100']) and \
            (df.iloc[cnt + 1]['EMA100'] < df.iloc[cnt]['Close']) and (
            df.iloc[cnt + 2]['EMA100'] > df.iloc[cnt + 1]['Close']):
        EMA100sellDate += df.iloc[cnt]['Date'] + ', '

    # Bollinger Bands
    if (df.iloc[cnt - 1]['Close'] > df.iloc[cnt]['LowerBand']) and (df.iloc[cnt]['Close'] < df.iloc[cnt]['LowerBand']):
        BBbuyDate += df.iloc[cnt]['Date'] + ', '
    if (df.iloc[cnt - 1]['Close'] < df.iloc[cnt - 1]['UpperBand']) and (
            df.iloc[cnt]['Close'] > df.iloc[cnt]['UpperBand']):
        BBsellDate += df.iloc[cnt]['Date'] + ', '

    # MACD
    if (df.iloc[cnt]['DIF'] > df.iloc[cnt]['DEA']) and (df.iloc[cnt - 1]['DIF'] < df.iloc[cnt - 1]['DEA']) and (
            df.iloc[cnt]['MACD'] > 0):
        MACDbuyDate += df.iloc[cnt]['Date'] + ', '
    if (df.iloc[cnt]['DIF'] < df.iloc[cnt]['DEA']) and (df.iloc[cnt - 1]['DIF'] > df.iloc[cnt - 1]['DEA']) and (
            df.iloc[cnt]['MACD'] < df.iloc[cnt - 1]['MACD']):
        MACDsellDate += df.iloc[cnt]['Date'] + ', '

    # DMI
    if (df.iloc[cnt - 1]['ADX'] < 25) and (df.iloc[cnt]['ADX'] > 25) and (
            df.iloc[cnt]['Plus_di'] > df.iloc[cnt]['Minus_di']):
        DMIbuyDate += df.iloc[cnt]['Date'] + ', '
    if (df.iloc[cnt - 1]['ADX'] < 25) and (df.iloc[cnt]['ADX'] > 25) and (
            df.iloc[cnt]['Plus_di'] < df.iloc[cnt]['Minus_di']):
        DMIsellDate += df.iloc[cnt]['Date'] + ', '

    # BIAS
    if df.iloc[cnt]['BIAS12'] <= -7:
        BIASbuyDate += df.iloc[cnt]['Date'] + ', '
    elif (df.iloc[cnt]['BIAS6'] > df.iloc[cnt]['BIAS24']) and (df.iloc[cnt - 1]['BIAS6'] < df.iloc[cnt - 1]['BIAS24']):
        BIASbuyDate += df.iloc[cnt]['Date'] + ', '

    if df.iloc[cnt]['BIAS12'] >= 7:
        BIASsellDate += df.iloc[cnt]['Date'] + ', '
    elif (df.iloc[cnt]['BIAS6'] < df.iloc[cnt]['BIAS24']) and (df.iloc[cnt - 1]['BIAS6'] > df.iloc[cnt - 1]['BIAS24']):
        BIASsellDate += df.iloc[cnt]['Date'] + ', '

    # KDJ
    if (df.iloc[cnt]['J'] < 10) and (df.iloc[cnt - 1]['J'] > 10):
        KDJbuyDate += df.iloc[cnt]['Date'] + ', '
    elif (df.iloc[cnt]['K'] > df.iloc[cnt]['D']) and (df.iloc[cnt - 1]['D'] > df.iloc[cnt - 1]['K']) and (
            df.iloc[cnt]['K'] < 20) and (df.iloc[cnt]['D'] < 20):
        KDJbuyDate += df.iloc[cnt]['Date'] + ', '

    if (df.iloc[cnt]['J'] > 100) and (df.iloc[cnt - 1]['J'] < 100):
        KDJsellDate += df.iloc[cnt]['Date'] + ', '
    elif (df.iloc[cnt]['K'] < df.iloc[cnt]['D']) and (df.iloc[cnt - 1]['D'] < df.iloc[cnt - 1]['K']) and (
            df.iloc[cnt]['K'] > 80) and (df.iloc[cnt]['D'] > 80):
        KDJsellDate += df.iloc[cnt]['Date'] + ', '

    # RSI
    if df.iloc[cnt]['RSI6'] < 20:
        if (df.iloc[cnt]['RSI6'] > df.iloc[cnt]['RSI12']) and (df.iloc[cnt - 1]['RSI6'] < df.iloc[cnt - 1]['RSI12']):
            RSIbuyDate += df.iloc[cnt]['Date'] + ', '
        elif (df.iloc[cnt]['RSI6'] > df.iloc[cnt]['RSI24']) and (df.iloc[cnt - 1]['RSI6'] < df.iloc[cnt - 1]['RSI24']):
            RSIbuyDate += df.iloc[cnt]['Date'] + ', '

    if df.iloc[cnt]['RSI6'] > 80:
        if (df.iloc[cnt]['RSI6'] < df.iloc[cnt]['RSI12']) and (df.iloc[cnt - 1]['RSI6'] > df.iloc[cnt - 1]['RSI12']):
            RSIsellDate += df.iloc[cnt]['Date'] + ', '
        elif (df.iloc[cnt]['RSI6'] < df.iloc[cnt]['RSI24']) and (df.iloc[cnt - 1]['RSI6'] > df.iloc[cnt - 1]['RSI24']):
            RSIsellDate += df.iloc[cnt]['Date'] + ', '

# ###################################
# ### Develop X_ & y_train & test data
length_df = len(df)
split_ratio = 0.8  # %80 train + %20 test
length_train = round(length_df * split_ratio)
length_test = length_df - length_train
test_start = df.iloc[length_train].name
training_set = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_set)
# Separating the data
training_set = df['Close'].iloc[:test_start].values
test_set = df['Close'].iloc[test_start:].values

prediction_days = 50
X_train, y_train = [], []
for i in range(prediction_days, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - prediction_days: i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Pre-processing the data
dataset_total = pd.concat((df["Close"].iloc[:test_start], df["Close"].iloc[test_start:]), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_set) - prediction_days:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
# Predict the values
X_test = []
for i in range(prediction_days,len(inputs)):
    X_test.append(inputs[i-prediction_days:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

# ### Evaluate & print metrics
def Calculate_print_metrics(test,predict):
    mape = mean_absolute_percentage_error(test, predict)
    rmse = mean_squared_error(test, predict)
    mae = mean_absolute_error(test, predict)
    # Print the evaluation metrics and directional accuracy
    st.write(f'Mean Absolute Percentage Error : {round(mape,4)}')
    st.write(f'Root Mean Squared Error : {round(rmse,4)}')
    st.write(f'Mean Absolute Error : {round(mae, 4)}')

# ###################################
# Print Current Date & Stock Price
st.write(f'Close Price of Date {df.iloc[-1]['Date']} is {round(df.iloc[-1]['Close'],2)}')
Last_Close_Price = df.iloc[(len(df)-1)]['Close']
# Plot Charts
# EMA Chart
st.subheader('Close Prices w/ Exponential Moving Average (Trend)')
lines_chart1 = px.line(df, x="Date", y=["Close", "EMA10", "EMA50", "EMA100"],
                           color_discrete_map={'Close': 'goldenrod','EMA10': 'blue','EMA50': 'green',
                                               'EMA100': 'purple'})
st.plotly_chart(lines_chart1)
st.write('EMA10  Buy Signal  : ', EMA10buyDate)
st.write('EMA10  Sell Signal : ', EMA10sellDate)
st.write('EMA50  Buy Signal  : ', EMA50buyDate)
st.write('EMA50  Sell Signal : ', EMA50sellDate)
st.write('EMA100 Buy Signal  : ', EMA100buyDate)
st.write('EMA100 Sell Signal : ', EMA100sellDate)

# Candlestick Chart
st.subheader('Candlestick Chart')
candlestick = go.Candlestick(x=df['Date'],
                             open=df['Open'], high=df['High'], low=df['Low'],
                             close=df['Close'], name='Candlestick')
candlestick_layout = go.Layout()
candlestick_fig = go.Figure(data=candlestick, layout=candlestick_layout)
st.plotly_chart(candlestick_fig)

# Volume Chart
bar_graph0 = px.bar(df, x=df['Date'],y=df['Volume']/1000000)
st.subheader('Volume')
st.plotly_chart(bar_graph0)

# Display Technical Indicators
if MACD_DMI :
# MACD Chart
    bar_graph1 = px.bar(df, x='Date',y='MACD')
    st.subheader('Moving Average Convergence Divergence (Trend)')
    st.plotly_chart(bar_graph1)
    st.write('MACD Buy Signal  : ', MACDbuyDate)
    st.write('MACD Sell Signal : ', MACDsellDate)
# DMI Chart
    lines_chart4 = px.line(df, x='Date',y=['Plus_di','Minus_di','ADX'])
    st.subheader('Directional Movement Index (Trend)')
    st.plotly_chart(lines_chart4)
    st.write('DMI  Buy Signal  : ', DMIbuyDate)
    st.write('DMI  Sell Signal : ', DMIsellDate)

if RSI_KDJ :
# RSI Chart
    lines_chart6 = px.line(df, x='Date',y=['RSI6','RSI12','RSI24'])
    st.subheader('Relative Strength Index (Momentum)')
    st.plotly_chart(lines_chart6)
    st.write('RSI  Buy Signal  : ', RSIbuyDate)
    st.write('RSI  Sell Signal : ', RSIsellDate)
# KDJ Chart
    lines_chart5 = px.line(df, x='Date',y=['K','D'])
    st.subheader('KDJ (Momentum)')
    st.plotly_chart(lines_chart5)
    st.write('KDJ  Buy Signal  : ', KDJbuyDate)
    st.write('KDJ  Sell Signal : ', KDJsellDate)

if BB_BIAS :
# BB Chart
    st.subheader('Bollinger Bands (Volatility)')
    lines_chart2 = px.line(df, x='Date', y=['Close', 'UpperBand', 'MiddleBand', 'LowerBand'],
                           color_discrete_map={'Close': 'goldenrod', 'UpperBand': 'blue', 'MiddleBand': 'green',
                                               'LowerBand': 'purple'})
    st.plotly_chart(lines_chart2)
    st.write('BB   Buy Signal    : ', BBbuyDate)
    st.write('BB   Sell Signal   : ', BBsellDate)
# BIAS Chart
    lines_chart7 = px.line(df, x='Date',y=['BIAS6','BIAS12','BIAS24'])
    st.subheader('BIAS (Volatility)')
    st.plotly_chart(lines_chart7)
    st.write('BIAS Buy Signal  : ', BIASbuyDate)
    st.write('BIAS Sell Signal : ', BIASsellDate)

# Resistance & Support Levels
st.subheader('Resistance & Support Levels')
st.write('Resistance Levels : ', ResistDate)
st.write('Support Levels : ', SupportDate)

# ###################################
# Prediction Models

# Predict the next 1~20 days' prices
# Select Exchange Calendars
import pandas_market_calendars as mcal

# Markets Calendar handling
hk_calendar = mcal.get_calendar('HKEX')
us_calendar = mcal.get_calendar('NYSE')
calendar = hk_calendar if 'hk' in code.lower() else us_calendar
valid_days = calendar.valid_days(start_date=pd.to_datetime(df.iloc[-1]['Date']),
                                 end_date=pd.to_datetime(df.iloc[-1]['Date']) + pd.DateOffset(days=365))
df_ticker = yf.download(code, start=valid_days[0], period='max')
if isinstance(df_ticker.columns, pd.MultiIndex):
    df_ticker = df_ticker.droplevel('Ticker', axis=1)
df_ticker = df_ticker[['Close']].head(pred_days)
df_ticker.rename(columns={'Close': 'Actual Close'}, inplace=True)
df_ticker.index = df_ticker.index.strftime('%Y-%m-%d')

if New_XGB or Rel_XGB:
    st.subheader('eXtreme Gradient Boosting Model (Price Changes & Lag Features')

    # 1. Create a local copy to avoid interfering with your deep learning models
    df_xgb = df.copy()

    # 2. Add Lag Features (Giving the tree a short-term memory window)
    df_xgb['Close_Lag1'] = df_xgb['Close'].shift(1)
    df_xgb['Close_Lag2'] = df_xgb['Close'].shift(2)
    df_xgb['Close_Lag3'] = df_xgb['Close'].shift(3)

    # 3. Create Stationary Target: Predict the percentage return of the NEXT day
    df_xgb['Next_Return'] = df_xgb['Close'].pct_change().shift(-1)

    # Clean up empty NaN rows generated by our shift operations
    df_xgb = df_xgb.dropna(subset=['Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Next_Return']).copy()

    # Update feature list to include our explicit lag columns
    features_list = ['Open', 'High', 'Low', 'Volume', 'EMA10', 'EMA50',
                     'UpperBand', 'MiddleBand', 'LowerBand', 'MACD',
                     'ADX', 'K', 'D', 'BIAS6', 'BIAS12', 'BIAS24', 'RSI6', 'RSI12', 'RSI24',
                     'Close_Lag1', 'Close_Lag2', 'Close_Lag3']

    X = df_xgb[features_list]
    y = df_xgb['Next_Return']  # Target is now stationary percentage change

    # Store absolute closes to convert predictions back to dollar amounts later
    actual_closes = df_xgb['Close'].values

    # Split the data sequentially (80/20)
    X_train_XGB, X_test_XGB, y_train_XGB, y_test_XGB = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    split_idx = len(X_train_XGB)
    test_actual_closes = actual_closes[split_idx:]

    if New_XGB:
        XGB_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
        XGB_model.fit(X_train_XGB, y_train_XGB)
        XGB_model.save_model(f"{code}_XGB_model.json")
    else:
        XGB_model = xgb.XGBRegressor()
        XGB_model.load_model(f"{code}_XGB_model.json")

    # Predict the expected NEXT-DAY percentage returns
    y_pred_returns = XGB_model.predict(X_test_XGB)

    # CONVERSION: Turn stationary returns back into absolute closing prices!
    # Formula: Tomorrow Price = Today Price * (1 + Predicted Return)
    y_pred_price = test_actual_closes * (1 + y_pred_returns)
    y_test_prices = test_actual_closes * (1 + y_test_XGB.values)

    # Evaluate using absolute values to align with your dashboard tables
    Calculate_print_metrics(y_test_prices, y_pred_price)

    # 4. Generate multi-step forward forecasting using cumulative rolling drift
    last_known_close = df['Close'].iloc[-1]
    future_preds = []
    latest_features = X.iloc[-1:].copy()

    for i in range(1, pred_days + 1):
        pred_return = XGB_model.predict(latest_features)[0]
        next_price = last_known_close * (1 + pred_return)
        future_preds.append(next_price)

        # Roll forward variables for next step
        last_known_close = next_price
        latest_features['Close_Lag3'] = latest_features['Close_Lag2']
        latest_features['Close_Lag2'] = latest_features['Close_Lag1']
        latest_features['Close_Lag1'] = next_price

    # Create DataFrame with index starting from 1
    future_XGB_df = pd.DataFrame(
        {"Date": [valid_days[i - 1].strftime('%Y-%m-%d') for i in range(1, pred_days + 1)],
         "XGB Predict": future_preds},
        index=range(1, pred_days + 1))
    future_XGB_df.set_index('Date', inplace=True)

    # Merge the two DataFrames on the index (Date)
    combo_XGB_df = future_XGB_df.join(df_ticker, how='outer')
    combo_XGB_df[['XGB Predict', 'Actual Close']] = combo_XGB_df[['XGB Predict', 'Actual Close']].round(2)
    st.dataframe(combo_XGB_df)
    combo_XGB_df['Abs_Err'] = abs(combo_XGB_df['XGB Predict'] - combo_XGB_df['Actual Close']).round(4)
    combo_XGB_df['Pct %_Err'] = (100 * combo_XGB_df['Abs_Err'] / combo_XGB_df['Actual Close']).round(4)
    combo_XGB_df['MAPE %-1D'] = combo_XGB_df['Pct %_Err'][:1].mean().round(4)
    combo_XGB_df['MAPE %-3D'] = combo_XGB_df['Pct %_Err'][:3].mean().round(4)
    combo_XGB_df['MAPE %-5D'] = combo_XGB_df['Pct %_Err'][:5].mean().round(4)
    combo_XGB_df['MAPE %-10D'] = combo_XGB_df['Pct %_Err'][:10].mean().round(4)
    combo_XGB_df['MAPE %-20D'] = combo_XGB_df['Pct %_Err'][:20].mean().round(4)
    st.dataframe(combo_XGB_df.iloc[0][4:])
    st.session_state['XGB_mapes'] = [combo_XGB_df['MAPE %-1D'].iloc[0], combo_XGB_df['MAPE %-3D'].iloc[0],
                                     combo_XGB_df['MAPE %-5D'].iloc[0], combo_XGB_df['MAPE %-10D'].iloc[0],
                                     combo_XGB_df['MAPE %-20D'].iloc[0]]

if New_GRU or Rel_GRU :
    st.subheader('Gated Recurrent Unit Model')
    if New_GRU:
        # 1. Initialize the sequential model container
        GRU_model = Sequential()
        # 2. ADD YOUR ACCELERATED GPU LAYERS HERE:
        GRU_model.add(GRU(64,
                          return_sequences=True, activation='tanh',
                          recurrent_activation='sigmoid', use_bias=True,
                          input_shape=(X_train.shape[1], 1)))
        GRU_model.add(Dropout(0.2))
        GRU_model.add(GRU(32, activation='tanh', use_bias=True,
                          return_sequences=False, recurrent_activation='sigmoid'))
        GRU_model.add(Dropout(0.2))
        GRU_model.add(Dense(1))
        GRU_model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        GRU_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1,
                      callbacks=[early_stopping], verbose=0)
        GRU_model.save(f"{code}_GRU_model.keras")
    else:
        GRU_model = load_model(f"{code}_GRU_model.keras")

    pred_out = GRU_model.predict(X_test)
    Calculate_print_metrics(test_set, scaler.inverse_transform(pred_out))

    last_prices = df['Close'][-50:].values.reshape(-1, 1)
    last_prices_scaled = scaler.fit_transform(last_prices)
    predicted_prices=[]
    current_batch=last_prices_scaled[-50:].reshape(1,50,1)
    for i in range(pred_days):
        next_prediction=GRU_model.predict(current_batch)
        next_prediction_reshaped=next_prediction.reshape(1,1,1)
        current_batch=np.append(current_batch[:,1:,:],next_prediction_reshaped,axis=1)
        predicted_prices.append(float(scaler.inverse_transform(next_prediction)[0][0]))

    # Create DataFrame with index starting from 1
    future_GRU_df = pd.DataFrame({"Date": [valid_days[i-1].strftime('%Y-%m-%d') for i in range(1, pred_days + 1)],
                                  "GRU Predict": [predicted_prices[i-1] for i in range(1, pred_days + 1)]},
                                     index=range(1, pred_days + 1))
    future_GRU_df.set_index('Date', inplace=True)

    # Merge the two DataFrames on the index (Date)
    combo_GRU_df = future_GRU_df.join(df_ticker, how='outer')
    combo_GRU_df[['GRU Predict','Actual Close']] = combo_GRU_df[['GRU Predict','Actual Close']].round(2)
    st.dataframe(combo_GRU_df)
    combo_GRU_df['Abs_Err']=abs(combo_GRU_df['GRU Predict']-combo_GRU_df['Actual Close']).round(4)
    combo_GRU_df['Pct %_Err']=(100*combo_GRU_df['Abs_Err']/combo_GRU_df['Actual Close']).round(4)
    combo_GRU_df['MAPE %-1D']=combo_GRU_df['Pct %_Err'][:1].mean().round(4)
    combo_GRU_df['MAPE %-3D']=combo_GRU_df['Pct %_Err'][:3].mean().round(4)
    combo_GRU_df['MAPE %-5D']=combo_GRU_df['Pct %_Err'][:5].mean().round(4)
    combo_GRU_df['MAPE %-10D']=combo_GRU_df['Pct %_Err'][:10].mean().round(4)
    combo_GRU_df['MAPE %-20D']=combo_GRU_df['Pct %_Err'][:20].mean().round(4)
    st.dataframe(combo_GRU_df.iloc[0][4:])
    st.session_state['GRU_mapes'] = [combo_GRU_df['MAPE %-1D'].iloc[0], combo_GRU_df['MAPE %-3D'].iloc[0],
                                     combo_GRU_df['MAPE %-5D'].iloc[0], combo_GRU_df['MAPE %-10D'].iloc[0],
                                     combo_GRU_df['MAPE %-20D'].iloc[0]]

if New_LSTM or Rel_LSTM:
    st.subheader('Long Short Term Memory Model')
    if New_LSTM:
        LSTM_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)])
        LSTM_model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        LSTM_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1, callbacks=[early_stopping],
                       verbose=0)
        LSTM_model.save(f"{code}_LSTM_model.keras")
    else:
        LSTM_model = load_model(f"{code}_LSTM_model.keras")

    pred_out = LSTM_model.predict(X_test)
    Calculate_print_metrics(test_set, scaler.inverse_transform(pred_out))

    last_prices = df['Close'][-50:].values.reshape(-1, 1)
    last_prices_scaled = scaler.fit_transform(last_prices)
    predicted_prices=[]
    current_batch=last_prices_scaled[-50:].reshape(1,50,1)
    for i in range(pred_days):
        next_prediction=LSTM_model.predict(current_batch)
        next_prediction_reshaped=next_prediction.reshape(1,1,1)
        current_batch=np.append(current_batch[:,1:,:],next_prediction_reshaped,axis=1)
        predicted_prices.append(float(scaler.inverse_transform(next_prediction)[0][0]))

    # Create DataFrame with index starting from 1
    future_LSTM_df = pd.DataFrame({"Date": [valid_days[i-1].strftime('%Y-%m-%d') for i in range(1, pred_days + 1)],
                                  "LSTM Predict": [predicted_prices[i-1] for i in range(1, pred_days + 1)]},
                                     index=range(1, pred_days + 1))
    future_LSTM_df.set_index('Date', inplace=True)

    # Merge the two DataFrames on the index (Date)
    combo_LSTM_df = future_LSTM_df.join(df_ticker, how='outer')
    combo_LSTM_df[['LSTM Predict','Actual Close']] = combo_LSTM_df[['LSTM Predict','Actual Close']].round(2)
    st.dataframe(combo_LSTM_df)
    combo_LSTM_df['Abs_Err']=abs(combo_LSTM_df['LSTM Predict']-combo_LSTM_df['Actual Close']).round(4)
    combo_LSTM_df['Pct %_Err']=(100*combo_LSTM_df['Abs_Err']/combo_LSTM_df['Actual Close']).round(4)
    combo_LSTM_df['MAPE %-1D']=combo_LSTM_df['Pct %_Err'][:1].mean().round(4)
    combo_LSTM_df['MAPE %-3D']=combo_LSTM_df['Pct %_Err'][:3].mean().round(4)
    combo_LSTM_df['MAPE %-5D']=combo_LSTM_df['Pct %_Err'][:5].mean().round(4)
    combo_LSTM_df['MAPE %-10D']=combo_LSTM_df['Pct %_Err'][:10].mean().round(4)
    combo_LSTM_df['MAPE %-20D']=combo_LSTM_df['Pct %_Err'][:20].mean().round(4)
    st.dataframe(combo_LSTM_df.iloc[0][4:])
    st.session_state['LSTM_mapes'] = [combo_LSTM_df['MAPE %-1D'].iloc[0], combo_LSTM_df['MAPE %-3D'].iloc[0],
                                      combo_LSTM_df['MAPE %-5D'].iloc[0], combo_LSTM_df['MAPE %-10D'].iloc[0],
                                      combo_LSTM_df['MAPE %-20D'].iloc[0]]

if New_LSTM_AM or Rel_LSTM_AM:
    st.subheader('Long Short Term Memory Model - Attention Model')
    if New_LSTM_AM:
        # CLEANED FIX: Direct non-destructive Attention usage
        inputs_layer = tf.keras.Input(shape=(X_train.shape[1], 1))
        lstm_out = LSTM(64, return_sequences=True)(inputs_layer)
        attn_out = AdditiveAttention()([lstm_out, lstm_out])
        flat = Flatten()(attn_out)
        outputs_layer = Dense(1)(flat)

        LSTM_AM_model = tf.keras.Model(inputs=inputs_layer, outputs=outputs_layer)
        LSTM_AM_model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        LSTM_AM_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1,
                          callbacks=[early_stopping], verbose=0)
        LSTM_AM_model.save(f"{code}_LSTM_AM_model.keras")
    else:
        LSTM_AM_model = load_model(f"{code}_LSTM_AM_model.keras")

    pred_out = LSTM_AM_model.predict(X_test)
    Calculate_print_metrics(test_set, scaler.inverse_transform(pred_out))

    last_prices = df['Close'][-50:].values.reshape(-1, 1)
    last_prices_scaled = scaler.fit_transform(last_prices)
    predicted_prices=[]
    current_batch=last_prices_scaled.reshape(1,50,1)
    for i in range(pred_days):
        next_prediction=LSTM_AM_model.predict(current_batch)
        next_prediction_reshaped=next_prediction.reshape(1,1,1)
        current_batch=np.append(current_batch[:,1:,:],next_prediction_reshaped,axis=1)
        predicted_prices.append(float(scaler.inverse_transform(next_prediction)[0][0]))
    # Create DataFrame with index starting from 1
    future_LSTMAM_df = pd.DataFrame({"Date": [valid_days[i-1].strftime('%Y-%m-%d') for i in range(1, pred_days + 1)],
                                  "LSTM-AM Predict": [predicted_prices[i-1] for i in range(1, pred_days + 1)]},
                                     index=range(1, pred_days + 1))
    future_LSTMAM_df.set_index('Date', inplace=True)

    # Merge the two DataFrames on the index (Date)
    combo_LSTMAM_df = future_LSTMAM_df.join(df_ticker, how='outer')
    combo_LSTMAM_df[['LSTM-AM Predict','Actual Close']] = combo_LSTMAM_df[['LSTM-AM Predict','Actual Close']].round(2)
    st.dataframe(combo_LSTMAM_df)
    combo_LSTMAM_df['Abs_Err']=abs(combo_LSTMAM_df['LSTM-AM Predict']-combo_LSTMAM_df['Actual Close']).round(4)
    combo_LSTMAM_df['Pct %_Err']=(100*combo_LSTMAM_df['Abs_Err']/combo_LSTMAM_df['Actual Close']).round(4)
    combo_LSTMAM_df['MAPE %-1D']=combo_LSTMAM_df['Pct %_Err'][:1].mean().round(4)
    combo_LSTMAM_df['MAPE %-3D']=combo_LSTMAM_df['Pct %_Err'][:3].mean().round(4)
    combo_LSTMAM_df['MAPE %-5D']=combo_LSTMAM_df['Pct %_Err'][:5].mean().round(4)
    combo_LSTMAM_df['MAPE %-10D']=combo_LSTMAM_df['Pct %_Err'][:10].mean().round(4)
    combo_LSTMAM_df['MAPE %-20D']=combo_LSTMAM_df['Pct %_Err'][:20].mean().round(4)
    st.dataframe(combo_LSTMAM_df.iloc[0][4:])
    st.session_state['LSTMAM_mapes'] = [combo_LSTMAM_df['MAPE %-1D'].iloc[0], combo_LSTMAM_df['MAPE %-3D'].iloc[0],
                                        combo_LSTMAM_df['MAPE %-5D'].iloc[0], combo_LSTMAM_df['MAPE %-10D'].iloc[0],
                                        combo_LSTMAM_df['MAPE %-20D'].iloc[0]]

if New_LSTM_FEAT or Rel_LSTM_FEAT:
    st.subheader('Long Short Term Memory - Features (Tech. Indicators) Model')
    # FIXED: Place 'Close' as index 0 so that predictions align properly in your future loop
    FEATURES = ['Close', 'High', 'Low', 'Open', 'EMA10', 'EMA50', 'UpperBand', 'LowerBand']
    # Secure row configurations clean of indicator warm-up windows
    df_features = df.dropna(subset=FEATURES).copy()
    # -------------------------------------------------------------
    # SAFE SCALING STRUCTURE & DATA PARTITION
    # -------------------------------------------------------------
    # Use standard MinMaxScaler to optimize internal tanh/sigmoid neural gates
    scaler_feat = MinMaxScaler(feature_range=(0, 1))
    np_data = scaler_feat.fit_transform(df_features[FEATURES])

    # Dedicated isolated single-column scaler for cleanly decoding inverse predictions
    scaler_pred = MinMaxScaler(feature_range=(0, 1))
    np_Close_scaled = scaler_pred.fit_transform(df_features[['Close']])

    prediction_days = 50

    def partition_dataset(lookback, data):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i, :])  # Window of all feature shapes
            y.append(data[i, 0])  # Target is index 0 ('Close')
        return np.array(X), np.array(y)

    # Split the scaled dataset sequentially using your existing clean training anchors
    train_data = np_data[:length_train, :]
    test_data = np_data[length_train - prediction_days:, :]
    X_train_F, y_train_F = partition_dataset(prediction_days, train_data)
    X_test_F, y_test_F = partition_dataset(prediction_days, test_data)

    # -------------------------------------------------------------
    # REFACTORED NEURAL ARCHITECTURE (FASTER & ACCURATE)
    # -------------------------------------------------------------
    if New_LSTM_FEAT:
        # FIXED: Removed the 350-neuron bottleneck. Switched to an efficient 64/32 stacking structure
        LSTM_FEAT_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train_F.shape[1], X_train_F.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='tanh'),
            Dense(1)])

        LSTM_FEAT_model.compile(optimizer='adam', loss='mean_squared_error')

        # Streamline validation and allow EarlyStopping to interrupt training early
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        LSTM_FEAT_model.fit(
            X_train_F, y_train_F,
            batch_size=32,
            epochs=25,  # Aligned with your standard model epoch limits
            callbacks=[early_stopping],
            shuffle=False,  # Maintain historical sequence tracking
            validation_data=(X_test_F, y_test_F),
            verbose=0)
        LSTM_FEAT_model.save(f"{code}_LSTM_FEAT_model.keras")

    if Rel_LSTM_FEAT:
        LSTM_FEAT_model = load_model(f"{code}_LSTM_FEAT_model.keras")

    # -------------------------------------------------------------
    # METRIC PRODUCTION & FUTURE TESTING HORIZONS
    # -------------------------------------------------------------
    y_pred_scaled = LSTM_FEAT_model.predict(X_test_F, verbose=0)
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test_F.reshape(-1, 1))

    Calculate_print_metrics(y_test_unscaled, y_pred)

    # Autoregressive multi-step projection calculation window
    last_prices_scaled = test_data[-prediction_days:]
    predicted_prices = []
    current_batch = last_prices_scaled.reshape(1, prediction_days, len(FEATURES))

    for i in range(pred_days):
        next_prediction = LSTM_FEAT_model.predict(current_batch, verbose=0)  # Shape (1, 1)
        # Formulate next sliding frame row
        last_row = current_batch[:, -1, :].copy()
        last_row[0, 0] = next_prediction[0, 0]  # Safely inserts the prediction into 'Close'
        next_row_reshaped = last_row.reshape(1, 1, len(FEATURES))
        current_batch = np.append(current_batch[:, 1:, :], next_row_reshaped, axis=1)
        # Decode and log the price prediction
        decoded_price = float(scaler_pred.inverse_transform(next_prediction)[0][0])
        predicted_prices.append(decoded_price)

    # Create DataFrame with index starting from 1
    future_LSTMFEAT_df = pd.DataFrame(
        {"Date": [valid_days[i - 1].strftime('%Y-%m-%d') for i in range(1, pred_days + 1)],
         "LSTM-FEAT Predict": [predicted_prices[i - 1] for i in range(1, pred_days + 1)]},
        index=range(1, pred_days + 1))
    future_LSTMFEAT_df.set_index('Date', inplace=True)

    # Merge performance summaries onto the primary UI dataframes
    combo_LSTMFEAT_df = future_LSTMFEAT_df.join(df_ticker, how='outer')
    combo_LSTMFEAT_df[['LSTM-FEAT Predict', 'Actual Close']] = combo_LSTMFEAT_df[
        ['LSTM-FEAT Predict', 'Actual Close']].round(2)
    st.dataframe(combo_LSTMFEAT_df)

    combo_LSTMFEAT_df['Abs_Err'] = abs(
        combo_LSTMFEAT_df['LSTM-FEAT Predict'] - combo_LSTMFEAT_df['Actual Close']).round(4)
    combo_LSTMFEAT_df['Pct %_Err'] = (100 * combo_LSTMFEAT_df['Abs_Err'] / combo_LSTMFEAT_df['Actual Close']).round(4)
    combo_LSTMFEAT_df['MAPE %-1D'] = combo_LSTMFEAT_df['Pct %_Err'].iloc[:1].mean().round(4)
    combo_LSTMFEAT_df['MAPE %-3D'] = combo_LSTMFEAT_df['Pct %_Err'].iloc[:3].mean().round(4)
    combo_LSTMFEAT_df['MAPE %-5D'] = combo_LSTMFEAT_df['Pct %_Err'].iloc[:5].mean().round(4)
    combo_LSTMFEAT_df['MAPE %-10D'] = combo_LSTMFEAT_df['Pct %_Err'].iloc[:10].mean().round(4)
    combo_LSTMFEAT_df['MAPE %-20D'] = combo_LSTMFEAT_df['Pct %_Err'].iloc[:20].mean().round(4)
    st.dataframe(combo_LSTMFEAT_df.iloc[0][4:])
    st.session_state['LSTMFEAT_mapes'] = [combo_LSTMFEAT_df['MAPE %-1D'].iloc[0],
                                          combo_LSTMFEAT_df['MAPE %-3D'].iloc[0],
                                          combo_LSTMFEAT_df['MAPE %-5D'].iloc[0],
                                          combo_LSTMFEAT_df['MAPE %-10D'].iloc[0],
                                          combo_LSTMFEAT_df['MAPE %-20D'].iloc[0]]

# #############################################################
# MULTI-MODEL PERFORMANCE COMPARISON SUMMARY TABLE
# #############################################################
st.markdown("---")
st.subheader("Mean Absolute Percentage Error Summary (MAPE %)")
# Define the matrix row placeholders matching the Excel framework layout
summary_data = {}
intervals = ["MAPE %-1D", "MAPE %-3D", "MAPE %-5D", "MAPE %-10D", "MAPE %-20D"]
# Fetch saved session values if the corresponding checkbox was selected during the calculation pass
if 'XGB_mapes' in st.session_state and (New_XGB or Rel_XGB):
    summary_data["XGB"] = st.session_state['XGB_mapes']
else:
    summary_data["XGB"] = [np.nan] * 5
if 'GRU_mapes' in st.session_state and (New_GRU or Rel_GRU):
    summary_data["GRU"] = st.session_state['GRU_mapes']
else:
    summary_data["GRU"] = [np.nan] * 5
if 'LSTM_mapes' in st.session_state and (New_LSTM or Rel_LSTM):
    summary_data["LSTM"] = st.session_state['LSTM_mapes']
else:
    summary_data["LSTM"] = [np.nan] * 5
if 'LSTMAM_mapes' in st.session_state and (New_LSTM_AM or Rel_LSTM_AM):
    summary_data["LSTM-Attention"] = st.session_state['LSTMAM_mapes']
else:
    summary_data["LSTM-Attention"] = [np.nan] * 5
if 'LSTMFEAT_mapes' in st.session_state and (New_LSTM_FEAT or Rel_LSTM_FEAT):
    summary_data["LSTM-Features"] = st.session_state['LSTMFEAT_mapes']
else:
    summary_data["LSTM-Features"] = [np.nan] * 5
# Build the structural comparison DataFrame
df_summary = pd.DataFrame(summary_data, index=intervals)
# Transpose the matrix to exactly replicate your requested Excel representation (Models as rows, Horizons as columns)
df_summary_transposed = df_summary.T
# Render the formatted dataframe cleanly to the Streamlit UI view pipeline
st.dataframe(df_summary_transposed.style.format(precision=4, na_rep="-").highlight_min(axis=0, color="lightgreen"))
st.caption("*Note: The model achieving the lowest prediction error is automatically highlighted in green.")

st.text("")
# ###################################
st.subheader('Full Stock Data')
st.dataframe(df)
finish_time = datetime.datetime.now()
st.write("Start Time :",start_time,"        ","Finish Time :",finish_time)
st.write("Elapsed Time :",finish_time-start_time)
st.text("")
st.text("Please note that this program is for informational purposes only and should not be taken as financial advice.")
st.text("We do not bear responsibility for any trading decisions made based on this program.")
st.text("Users are advised to conduct their own research or consult with a qualified financial professional before making any investment decisions.")

