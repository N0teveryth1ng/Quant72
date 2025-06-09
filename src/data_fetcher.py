import yfinance as yf
import pandas as pd
# from datetime import datetime
import os


# aapl data fetcher
def data_fetch(ticker='AAPL'):
    data = yf.download(ticker, start='2022-01-01', end='2024-05-31', auto_adjust=True)

    # ✅ Flatten multi-index if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index.name = 'Date'
    os.makedirs("src/data/raw", exist_ok=True)
    data.to_csv(f"src/data/raw/{ticker}_stock.csv")

    return data
#
# # amzn data fetcher
# def data_fetch_amzn(ticker='AMZN'):
#     data = yf.download(ticker, start='2022-01-01', end='2024-05-31', auto_adjust=True)
#
#     if isinstance(data.columns, pd.MultiIndex):
#         data.columns = data.columns.get_level_values(0)
#         # data.columns = [f"{col[0]}_{col[1]}" for col in data.columns]
#
#     data.index.name = 'Date'
#     os.makedirs("src/data/raw", exist_ok=True)
#     data.to_csv(f"src/data/raw/{ticker}_stock.csv")
#
#     return data



if __name__ == "__main__":
   data_fetch('AAPL')
   # data_fetch_amzn('AMZN')
