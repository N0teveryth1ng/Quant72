import yfinance as yf
import pandas as pd
# from datetime import datetime
import logging
import os

from streamlit import exception

os.makedirs("logs", exist_ok=True) # ensures log folder exists
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/data_fetcher.log"),
        logging.StreamHandler()
    ]
)

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



# Tests
if __name__ == "__main__":
   data_fetch('AAPL')
