from data_fetcher import  *
from src.data_fetcher import data_fetch
from textblob import TextBlob


df = data_fetch('AAPL')
text =  TextBlob(df)
print(text.sentiment)


