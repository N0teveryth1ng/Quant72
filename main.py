# MAIN.LOG - Main Terminus for all over testing [ROUND-OFF]

import logging
import os
from operator import index

import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

from src.data_fetcher import data_fetch
from sklearn.model_selection import train_test_split
from src.preprocess import compute_rsi
from src.train import X_train, X_test, y_test
from tabulate import tabulate


# With XGB
def run_pipeline_2(X, y):
  try:
    logging.info('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    logging.info('Training XG-Boost Classifier...')
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Results
    logging.info(f"XGB - Train accuracy: {train_score:.4f} ")
    logging.info(f"XGB - Test accuracy: {test_score:.4f} ")

    return model

  except Exception as e:
      logging.error(f"Something went wrong: {e} ")
      return None


#   price_prediction pipeline with ARIMA for APPLE
def price_pred(time_series, test_size=10, order=(1,1,0)):
    try:
         logging.info('Splitting data . . . ')

         # validate input
         if len(time_series) <= test_size:
             raise ValueError(f"Time series too short ({len(time_series)}) for test size {test_size}")

         # split data
         train = time_series[:-test_size]
         test = time_series[-test_size:]
         logging.info(f"split complete: Train - {len(train)} days, Test - {len(test)} days")

         # Create and fit model
         logging.info(f'Fitting ARIMA {order} model ')
         model = ARIMA(train, order=order)
         model = model.fit()
         logging.info('Model Fitting Done')

         predictions = model.forecast(steps=test_size)
         logging.info(f"Predictions generated {len(predictions)} ")

         return  model, predictions, test

    except Exception as e:
        logging.error(f"Something went wrong: {e}")
        return None



# Price prediction with RSI
def apply_rsi():
    try:
        df = data_fetch('AAPL')
        logging.info("fetching data - [Implementing RSI]...")

        df = compute_rsi(df, window=14)

        table_df = df[['Close', 'RSI']].tail()
        print(tabulate(table_df, headers='keys', tablefmt='grid', showindex=False))

        os.makedirs("src/data", exist_ok=True)
        df.to_csv("src/data/AAPL_current.csv")


    except Exception as e:
        logging.error(f"Somthing went wrong: {e}")
        return None


# Predicting using MACD
from src.preprocess import compute_macd
def apply_macd():
    try:
        df = data_fetch('AAPL')
        logging.info("Fetching data - [Implementing MACD]...")

        df = compute_macd(df)
        table_df = df[['Close','MACD', 'Signal_Line']].tail()

        print(tabulate(table_df, headers='keys', tablefmt='grid', showindex=False))

        os.makedirs("src/data", exist_ok=True)
        df.to_csv("src/data/AAPL_macd.csv")
        return df


    except Exception as e:
        logging.error(f"Something went wrong {e}")
        return None


# Predicting LAG Returns --
from sklearn.ensemble import RandomForestClassifier
from src.preprocess import train_df, compute_lagReturns

def apply_LagRets():
  try:
    logging.info('Calculating Lag Returns...')

    X,y = compute_lagReturns(train_df)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42
   )

    # Train models
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Accuracy
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Results
    logging.info(f"RFC - Train accuracy: {train_score:.4f} ")
    logging.info(f"RFC - Test accuracy: {test_score:.4f} ")


  except Exception as e:
      logging.error(f"Something went wrong {e}")



# stochastic application
from src.preprocess import compute_stochastic

def apply_stocastic():
    try:
        df = data_fetch('AAPL')
        logging.info('Fetching data - [Stochastic Oscillator]')

        df = compute_stochastic(df)
        table_df = df[['Close', '%K', '%D']].tail()
        print(tabulate(table_df, headers='keys', tablefmt='grid', showindex=False))

        df.to_csv("src/data/AAPL_stochastic.csv")
        return df

    except Exception as e:
        logging.info(f" Something went wrong {e}")
        return None



#       none
def backtest_combined_strategy(df):
    df = df.copy()

    # Generate signal
    df['Signal'] = 0
    df.loc[(df['MACD'] > df['Signal_Line']) & (df['RSI'] < 60), 'Signal'] = 1  # Buy
    df.loc[(df['MACD'] < df['Signal_Line']) & (df['RSI'] > 40), 'Signal'] = -1  # Sell

    df['Position'] = df['Signal'].shift(1)
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Position'] * df['Market_Return']

    # Cumulative returns
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative_Strategy'], label='Strategy Returns', color='green')
    plt.plot(df.index, df['Cumulative_Market'], label='Market Returns', color='blue')
    plt.title('Combined MACD + RSI Strategy vs Market')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# For Tests
if __name__ == "__main__":
    df = data_fetch('AAPL')
    df = compute_rsi(df, window=14)
    df = compute_macd(df)


    apply_LagRets()
    apply_stocastic()

    backtest_combined_strategy(df)

    # run_pipeline_2() # calling XGB
    # apply_rsi()
    # apply_LagRets()
    # apply_macd()
    #
    # df_macd = apply_macd()  # ✅ store returned df
    # if df_macd is not None:
    #     backtest_macd(df_macd)  # ✅ pass it to backtest
    #
    # # Example usage
    # import pandas as pd
    # import numpy as np
    #
    # # Create sample data if no data passed
    # dates = pd.date_range(start='2023-01-01', periods=100)
    # prices = pd.Series(np.random.rand(100) * 100 + 150, index=dates)
    #
    # # Test the function
    # model, preds, test = price_pred(prices)
    #
    #
    # if model:  # Only if successful
    #     print(f"Test values: {test.values}")
    #     print(f"Predictions: {preds}")


















