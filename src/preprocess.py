import random
from src.data_fetcher import data_fetch, data_fetch_2024
import pandas as pd


# - - - - > AAPL data pre-process
apple_df = data_fetch(ticker='AAPL')

# historic data of apple
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
df['Target'] = df['CLose'].shift(-1) # tomorrow's future price [ Predict ]
prices = df['Close'].dropna() # price closed at

df.dropna(inplace=True) # drop missing vals

# train - test for - - - > [Prices]
train = prices[:-10]
test = prices[-10:]

# final features
X1 = df[['Prev_Close', '5_day_avg']]
y1 = df['Target']



