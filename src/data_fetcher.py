import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def data_fetch(ticker='AAPL'):
    data = yf.download(ticker, start='2022-01-01', end='2024-05-31', auto_adjust=True)

    # ✅ Flatten multi-index if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = 'Date'
    data.to_csv(f"src/data/raw/{ticker}_stock.csv")

    return data


if __name__ == "__main__":
   data_fetch('AAPL')
