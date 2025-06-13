import yfinance as yf
import pandas as pd
import datetime
import os

from dateutil.utils import today
from scipy.stats import loggamma_gen

from src.logger import get_logger

logging = get_logger("data_fetch", "logs/data_fetcher.log")

# apple data fetcher
def data_fetch(ticker='AAPL'):
  try:
    logging.info(f"data fetching for {ticker}...")
    data = yf.download(ticker, start='2022-01-01', end='2024-05-31', auto_adjust=True)

    # ✅ Flatten multi-index if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = 'Date'
    os.makedirs("src/data/raw", exist_ok=True)
    csv_path = f"src/data/raw/{ticker}_stock.csv"
    data.to_csv(csv_path)

    logging.info(f"Data saved to csv{csv_path}")
    return data

  except Exception as e:
    logging.error(f"something went wrong: {e}")
    return None





# data fetching of apple for price prediction - current
def data_fetch_2024(ticker='AAPL'):
   try:
     logging.info(f'Fetching current market data {ticker}. . .')
     data = yf.download(ticker,start='2023-04-01', end=datetime.date.today() ,auto_adjust=True)

     if isinstance(data.columns, pd.MultiIndex):
        data.columns  = data.columns.get_level_values(0)

     data.index.name = 'date'
     os.makedirs(f"src/data{ticker}_current.csv",exist_ok=True)
     csv_path = f"src/data{ticker}_current.csv"
     data.to_csv(csv_path)

     logging.info(f"Date saved to csv: {ticker}")
     return data

   except Exception as e:
      logging.error(f'Something went wrong: {e} ')

# Tests
if __name__ == "__main__":
   data_fetch('AAPL')
