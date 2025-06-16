# MAIN.LOG - Main Terminus for all over testing [ROUND-OFF]

import logging
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

# import src.data_fetcher
# from src.data_fetcher import data_fetch
# from src.preprocess import X, y
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from src.preprocess import train, test

# loads -
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error


# With XGB
def run_pipeline_2(X, y):
  try:
    logging.info('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    logging.info('Training XG-Boost Classifier...')
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logos')
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
         train = time_series[:test_size]
         test = time_series[test_size:]
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




# For Tests
if __name__ == "__main__":
    # run_pipeline_2(X, y) # calling XGB
    # price_pred() # Predicting future price

    # Example usage
    import pandas as pd
    import numpy as np

    # Create sample data if no data passed
    dates = pd.date_range(start='2023-01-01', periods=100)
    prices = pd.Series(np.random.rand(100) * 100 + 150, index=dates)

    # Test the function
    model, preds, test = price_pred(prices)

    if model:  # Only if successful
        print(f"Test values: {test.values}")
        print(f"Predictions: {preds}")








