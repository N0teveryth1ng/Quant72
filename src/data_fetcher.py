import yfinance as yf
from datetime import datetime
import os

# fetch data function
def data_fetch(ticker='TSLA',start='2023-01-01',end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    os.makedirs("data/raw", exist_ok=True)

    data = yf.download(ticker, start=start ,end=end)
    data.reset_index(inplace=True)

    csv_path = f"data/raw/{ticker}_stock.csv"
    data.to_csv(csv_path,index=False)
    print(f"data saved to {csv_path}")
    return data

if __name__ == "__main__":
   data_fetch('AAPL')
#
#
# # Test case
# ticker = 'MSFT'
# data = yf.download(ticker,auto_adjust=False)
# print(data.info())