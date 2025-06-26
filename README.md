# 📊 Quant72: Hybrid Technical Indicator-Based Trading Strategy

A Python-based end-to-end trading strategy toolkit combining machine learning and technical indicators like RSI, MACD, Stochastic Oscillator, ARIMA time-series forecasting, and lagged returns. Focused on signal generation, performance visualization, and backtesting.

---

## 🛠 Features

- 📈 **Technical Indicators**:  
  - Relative Strength Index (RSI)  
  - Moving Average Convergence Divergence (MACD)  
  - Stochastic Oscillator  

- 🧠 **Machine Learning**:  
  - XGBoost Classifier  
  - Random Forest on lagged returns  

- 🔁 **Time Series Forecasting**:  
  - ARIMA model for price prediction  

- 📊 **Backtesting**:
  - RSI + MACD combined strategy  
  - Stochastic Oscillator strategy  
  - Visualized vs. market returns  

---

## 🔁 Strategy Backtests

- Combined MACD + RSI crossover logic with signal-driven cumulative return visualization  
- Separate strategy backtesting for stochastic oscillator  
- Compare with actual market return performance  

---

## 🧪 Files & Modules

- `main.py`: Runs all strategy pipelines and backtests  
- `src/preprocess.py`: Computes all technical indicators  
- `src/data_fetcher.py`: Fetches and stores stock data via `yfinance`  
- `src/train.py`: Contains models and training utilities  

---

## 📦 Installation

```bash
git clone https://github.com/your-username/quant72.git
cd quant72

Tools Used: 
yfinance
pandas
numpy
scikit-learn
matplotlib
xgboost
statsmodels
tabulate
```

## 📄 Results

![image](https://github.com/user-attachments/assets/780ad72c-9b65-4539-96ef-e5aafecaabc7)



---

