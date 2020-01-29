"""
    trains a separate model on each of the first 5 stock ticker's prices for 
    the last 5 years and saves them
"""
import pickle
import pandas as pd
import datetime as dt
from MLTrader import MLTrader

#storing the symbols, starting and ending dates for training the models
tickers = list(pd.read_csv("yfinance_tickers.csv").iloc[:5,:].Symbol.values)
end_date =  dt.datetime.today() 
start_date = end_date - relativedelta(years=5)

#reading in adjusted closing prices
prices = pull_prices(symbol, sd, ed)

#normalizing the prices
ss = StandardScaler()
prices_norm = pd.DataFrame(ss.fit_transform(prices),index=prices.index, columns=[symbol])

#generating the indicators from the price data
trader = MLTrader(Ridge, n = 10)
trader.fit()
trader.save_learner(symbol)