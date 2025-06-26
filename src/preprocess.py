import logging
import random

from sympy import false
from ta.others import daily_return

from src.data_fetcher import data_fetch, data_fetch_2024


df = data_fetch('AAPL')
if df is None:
    raise ValueError("Data fetch failed")


# historic data of apple
apple_df = data_fetch(ticker='AAPL')
train_df = apple_df.loc['2023-01-01':'2024-03-31']
test_df = apple_df.loc['2024-04-01':]

# feature engineering
train_df['daily_return'] = train_df['Close'].pct_change()
train_df['volatility'] = train_df['daily_return'].rolling(window=5).std()
train_df['volatile'] = (train_df['volatility'] > 0.02).astype(int)

# fake sentiment checker
def fake_sentiment():
    return round(random.uniform(-1,1), 2)

train_df['sentiment'] = [fake_sentiment() for _ in range (len(train_df))]

# final feature prep
X = train_df[['daily_return', 'sentiment']].dropna()
y = train_df['volatile'].loc[X.index]







# Data preprocess & feature engineering for prediction - - - >
df = data_fetch_2024(ticker='AAPL')

# feature engineering
df['Prev_Close'] = df['Close'].shift(1) #yesterday's moving avg
df['5_day_avg'] = df['Close'].rolling(5).mean() # 5 days MA
df['Target'] = df['Close'].shift(-1) # tomorrow's future price [ Predict ]
prices = df['Close'].dropna() # price closed at

df.dropna(inplace=True) # drop missing vals

# train - test for - - - > [Prices]
train = prices[:-10]
test = prices[-10:]

# final features
X1 = df[['Prev_Close', '5_day_avg']]
y1 = df['Target']



# RSI function
def compute_rsi(df, window=14):
    Delta = df['Close'].diff()
    gain = Delta.clip(lower=0)
    loss = -Delta.clip(upper=0)

    # avg gain/loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # relative strength
    rs = avg_gain / avg_loss
    rsi = 100 - ( 100 / (1+rs))

    # New df['RSI']
    df['RSI'] = rsi
    return df


# MACD
def compute_macd(df, fast=12, slow=26, signal=9):
    try:
        df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['Signal_Line'] = df['Close'].ewm(span=signal, adjust=False).mean()
        return df

    except Exception as e:
        logging.error(f"MACD error: {e}")
        return df


# LAG Returns
def compute_lagReturns(df, window=14):

    # adding LAG
    df['lag_1'] = df['daily_return'].shift(1)
    df['lag_2'] = df['daily_return'].shift(2)
    df['lag_3'] = df['daily_return'].shift(3)

    X = df[['daily_return', 'sentiment','lag_1', 'lag_2', 'lag_3']].dropna()
    y = df['volatile'].loc[X.index]

    return X, y


# stochastic process
def compute_stochastic(df, k_window=14, d_window=3):
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()

    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(window=d_window).mean()

    return df





""" HI. THIS PROJECT IS INTEGRATED BY Wrick aka @CHARTER """











