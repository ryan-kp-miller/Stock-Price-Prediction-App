#importing dependencies
import pandas as pd
import numpy as np

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
    sma = prices.rolling(window=n, min_periods=n).mean().dropna()
    rolling_std = prices.rolling(window=n, min_periods=n).std().dropna()
    top_band = sma + 2*rolling_std
    bottom_band = sma - 2*rolling_std
    bb = (prices.iloc[n:,:] - bottom_band) / (top_band - bottom_band)
    bb = bb.replace([np.inf, -np.inf], np.nan).dropna() #replacing inf with NaN and dropping rows with NaN's
    bb = (bb - bb.mean()) / bb.std()  #normalizing the indicator
    return bb


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
    daily_ret.iloc[0,:] = 0
    vol = daily_ret.rolling(window=n, min_periods=n).std()
    vol = (vol - vol.mean()) / vol.std()  #normalizing the indicator
    return vol
