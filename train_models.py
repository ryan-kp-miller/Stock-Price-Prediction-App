"""
    trains a separate model on each of the first 5 stock ticker's prices for
    the last 5 years and saves them
"""
import pandas as pd
import datetime as dt
from MLTrader import MLTrader
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import Ridge

#storing the symbols, starting and ending dates for training the models
tickers = list(pd.read_csv("yfinance_tickers.csv").Symbol.values)
end_date =  dt.datetime.today()
start_date = end_date - relativedelta(years=5)

#looping through the stock symbols, training a model, and saving it
for symbol in tickers:
    print(symbol)
    #generating the indicators from the price data
    trader = MLTrader(Ridge, n = 10, kwargs={'alpha':0.001, 'random_state':0})
    trader.fit(symbol, sd=start_date, ed=end_date)
    trader.save_learner(symbol)
