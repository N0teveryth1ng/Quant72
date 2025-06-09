import random
from src.data_fetcher import data_fetch
# from src.data_fetcher import data_fetch_amzn


# - - - - > AAPL data pre-process
apple_df = data_fetch(ticker='AAPL')

    # historic data of apple
train_df = apple_df.loc['2023-01-01':'2024-03-31']
test_df = apple_df.loc['2024-04-01':]

    # feature engineer
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


#
#
# # - - - - > AMZN data pre process
# amzn_df = data_fetch_amzn(ticker='AMZN')
#
# # train and test
# train_amzn_df = amzn_df.loc['2024-01-01':'2024-01-31']
# test_amzn_df = amzn_df.loc['2024-01-01']
#
# # feature building
# train_amzn_df['daily_return'] = train_amzn_df['Close'].pct_change()
# train_amzn_df['volatility'] = train_amzn_df['daily_return'].rolling(window=5).std()
# train_amzn_df['volatile'] = (train_amzn_df['volatility'] > 0.02).astype(int)
#
#
# # fake sentimnet analysis
# def fake_sentiment_amzn():
#     return round(random.uniform(-1,1),2)
#
# train_amzn_df['sentiment'] = [fake_sentiment_amzn() for _ in range (len(train_amzn_df))]
#
# X1 = train_amzn_df[['daily_return', 'sentiment']].dropna()
# y1 = train_amzn_df['volatile'].loc[X1.index]