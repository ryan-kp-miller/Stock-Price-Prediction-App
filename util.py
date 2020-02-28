import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
import yfinance as yf


def pull_prices(symbol, sd, ed):
    """
        helper method for reading in and preprocessing the prices data

        inputs:
            symbol: string representing the stock symbol for trading
            sd:     string representing the date to start trading
            ed:     string representing the date to stop trading

        output:
            prices: dataframe containing the preprocessed daily price data
                    for the given stock
    """
    #reading in the stock data using util.py and removing nulls
    df = yf.download(symbol, start=sd, end=ed,
                     group_by="ticker", auto_adjust=True)
    prices = df.filter(items=['Close'],axis=1)
    prices.columns = [symbol]
    prices.fillna(method='ffill', inplace=True) #forward-filling missing prices
    prices.fillna(method='bfill', inplace=True) #back-filling missing prices
    return prices
    
def pull_prices_viz(symbols, period="5y"):
    """
        helper method for reading in and preprocessing the prices data

        inputs:
            symbols: string representing the stock symbols for trading
            period:  string representing the period of time to pull the stock 
                     price for (e.g. "1m" for the last month)

        output:
            prices:  dataframe containing the preprocessed daily price data
                     for the given stock
    """
    #reading in the stock data using util.py and removing nulls
    df = yf.download(symbols, period="5y", auto_adjust=True)
    
    #dropping "today's closing price" 
    #(at run-time, there typically won't be a closing price for today)
    df = df.iloc[:-1,:]
    
    #pushing date from the index to a column and filling NaNs
    prices = df['Close'].reset_index()
    prices.fillna(method='ffill', inplace=True) #forward-filling missing prices
    prices.fillna(method='bfill', inplace=True) #back-filling missing prices
    return prices
    

#function for normalizing and plotting the given data
def plot_winnings(df, plot_name, labels, long_list = [], short_list = []):
    #normalizing the data
    df_norm = df / df.iloc[0,:]
    #plotting the normalized data
    df_norm.plot(color=['g','r','b'])
    #looping over the portfolio values
    if len(long_list) > 0:
        for i in range(len(long_list)):
            plt.axvline(x=long_list[i], color='blue',linestyle='--')
    if len(short_list) > 0:
        for i in range(len(short_list)):
            plt.axvline(x=short_list[i], color='black',linestyle='--')
    #cleaning up the chart
    plt.legend(df_norm.columns)
    plt.xlabel("Time")
    plt.ylabel("Normalized Growth")
    plt.title(plot_name)
    plt.show()
