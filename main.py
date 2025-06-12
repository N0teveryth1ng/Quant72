import src.data_fetcher
from src.data_fetcher import data_fetch
from src.preprocess import X, y
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

# loads -
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# With RDF
def run_pipeline_1():
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Results
    print("RDF - Train accuracy:", model.score(X_train, y_train))
    print("RDF - Test accuracy:", model.score(X_test, y_test))



# With XGB
def run_pipeline_2():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Results
    print("XGB - Train accuracy:", model.score(X_train, y_train))
    print("XGB - Test accuracy:", model.score(X_test, y_test))





# for tests
if __name__ == "__main__":
    # data_fetch('AAPL')  # Ensure latest data
    # run_pipeline_2() # calling RDF
    # run_pipeline_1() # calling XGB
    arima_pipeline() # calling btc one
