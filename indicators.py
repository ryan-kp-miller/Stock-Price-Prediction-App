"""
Student Name: Ryan Miller
GT User ID: rmiller327
GT ID: 903461824
"""

#importing dependencies
import pandas as pd
import matplotlib.pyplot as plt

#author function
def author():
    return 'rmiller327'


def price_sma_ratio(prices, n):
    """
        function to calculate the price/n-day SMA (Simple Moving
        Average) ratio of the given stocks

        inputs:
            prices:      dataframe containing the prices of the
                         given stocks
            n:           integer representing the number of days to
                         use for calculating momentum
        output:
            price_sma:   dataframe containing the normalized price
                         over sma for each trading day
    """
    sma = prices.rolling(window=n, min_periods=n).mean()
    price_sma = prices / sma
    price_sma = (price_sma - price_sma.mean()) / price_sma.std()  #normalizing the indicator
    return price_sma


def price_sma_plot(prices, n):
    """
        function to create and save a plot of the normalized price, n-day SMA,
        and the price/n-day SMA ratio of the given stocks

        inputs:
            prices:      dataframe containing the prices of the
                         given stocks
            n:           integer representing the number of days to
                         use for calculating momentum
    """
    #calculating SMA and price/SMA
    sma = prices.rolling(window=n, min_periods=n).mean().fillna(method='ffill').fillna(method='bfill')
    price_sma = prices / sma
    #normalizing price, SMA, and price/SMA
    prices_norm = prices / prices.iloc[0,:]
    sma_norm = sma / sma.iloc[0,:]
    price_sma_norm = price_sma / price_sma.iloc[0,:]
    #plotting the normalized data
    prices_norm.JPM.plot(color=['g'])
    sma_norm.JPM.plot(color=['r'])
    price_sma_norm.JPM.plot(color=['b'])

    #cleaning up the chart
    labels = ['Prices','SMA','Prices/SMA']
    plot_name = "Price_SMA Indicator"
    plt.legend(labels)
    plt.xlabel("Time")
    plt.ylabel("Normalized Growth")
    plt.title(plot_name)
    plt.savefig(plot_name+".png")  #saving the plot with the specified name
    plt.clf()   #clearing the current figure



def bollinger_bands(prices, n):
    """
        function to calculate the Boelinger Bands of the given stocks
        using an n-day SMA

        inputs:
            price:    dataframe containing the prices of the
                      given stocks
            n:        integer representing the number of days to
                      use for calculating momentum
        output:
            bb:       dataframe containing the normalized boelinger
                      band % of the given stocks for each trading day
    """
    sma = prices.rolling(window=n, min_periods=n).mean()
    rolling_std = prices.rolling(window=n, min_periods=n).std()
    top_band = sma + 2*rolling_std
    bottom_band = sma - 2*rolling_std
    bb = (prices - bottom_band) / (top_band - bottom_band)
    bb = (bb - bb.mean()) / bb.std()  #normalizing the indicator
    return bb

def bb_plot(prices, n):
    """
        function to create and save a plot of the normalized price and Bollinger
        Band percentages of the given stocks

        inputs:
            prices:      dataframe containing the prices of the
                         given stocks
            n:           integer representing the number of days to
                         use for calculating momentum
    """
    #calculating Bollinger Bands Percentage
    sma = prices.rolling(window=n, min_periods=n).mean().fillna(method='ffill').fillna(method='bfill')
    rolling_std = prices.rolling(window=n, min_periods=n).std().fillna(method='ffill').fillna(method='bfill')
    top_band = sma + 2*rolling_std
    bottom_band = sma - 2*rolling_std
    bb = (prices - bottom_band) / (top_band - bottom_band)
    #normalizing price
    prices_norm = prices / prices.iloc[0,:]
    #plotting the normalized data
    prices_norm.JPM.plot(color=['g'])
    bb.JPM.plot(color=['r'])

    #cleaning up the chart
    labels = ['Prices','Bollinger Band Percentage']
    plot_name = "Bollinger Band Percentage Indicator"
    plt.legend(labels)
    plt.xlabel("Time")
    plt.ylabel("Normalized Growth")
    plt.title(plot_name)
    plt.savefig(plot_name+".png")  #saving the plot with the specified name
    plt.clf()   #clearing the current figure


def volatility(prices, n):
    """
        function to calculate the n-day std dev of the daily returns of the
        given stocks

        inputs:
            price:       dataframe containing the prices of the
                         given stocks
            n:           integer representing the number of days to
                         use for calculating momentum
        output:
            vol:         dataframe containing the normalized n-day
                         rolling volatility of price for each
                         trading day
    """
    daily_ret = prices/prices.shift(1) - 1
    daily_ret.ix[0,:] = 0
    vol = daily_ret.rolling(window=n, min_periods=n).std()
    vol = (vol - vol.mean()) / vol.std()  #normalizing the indicator
    return vol

def volatility_plot(prices, n):
    """
        function to create and save a plot of the normalized price and
        volatility of the given stocks

        inputs:
            prices:      dataframe containing the prices of the
                         given stocks
            n:           integer representing the number of days to
                         use for calculating momentum
    """
    #calculating volatility
    daily_ret = prices/prices.shift(1) - 1
    daily_ret.ix[0,:] = 0
    vol = daily_ret.rolling(window=n, min_periods=n).std().fillna(method='ffill').fillna(method='bfill')
    #normalizing price, SMA, and price/SMA
    prices_norm = prices / prices.iloc[0,:]
    vol_norm = vol / vol.iloc[0,:]
    #plotting the normalized data
    prices_norm.JPM.plot(color=['g'])
    vol_norm.JPM.plot(color=['r'])

    #cleaning up the chart
    labels = ['Prices','Volatility']
    plot_name = "Volatility Indicator"
    plt.legend(labels)
    plt.xlabel("Time")
    plt.ylabel("Normalized Growth")
    plt.title(plot_name)
    plt.savefig(plot_name+".png")  #saving the plot with the specified name
    plt.clf()   #clearing the current figure
