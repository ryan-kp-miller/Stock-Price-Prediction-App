"""MC2-P1: Market simulator.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Ryan Miller
GT User ID: rmiller327
GT ID: 903461824
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
import yfinance as yf


def author():
    return 'rmiller327'


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
    plt.savefig(plot_name+".png")  #saving the plot with the specified name
    plt.clf()   #clearing the current figure


#calculates average daily return and the standard deviation of daily returns
def port_stats(prices):
    cr = prices[-1]/prices[0] - 1
    daily_ret = prices/prices.shift(1) - 1
    adr = np.mean(daily_ret)
    sddr = np.std(daily_ret)
    return adr,sddr, cr


def compute_portvals(orders, start_val = 1000000, commission=9.95, impact=0.005):
    #retrieving parameters to pull data from orders df
    sd = orders.index.min()
    ed = orders.index.max()
    symbols = list(orders.Symbol.unique())

    #reading in adjusted close prices for the stocks in orders
    prices = yf.download(symbols[0], start=sd, end=ed,
                     group_by="ticker", auto_adjust=True)
    prices = prices.filter(items=['Close'],axis=1)
    prices.columns = [symbols[0]]
    prices.fillna(method='ffill', inplace=True) #forward-filling missing prices
    prices.fillna(method='bfill', inplace=True) #back-filling missing prices
    prices['Cash'] = 1  #setting value of cash to 1

    #creating list of dates the market was open for trading
    market_dates = [i.strftime('%Y-%m-%d') for i in prices.index]

    #creating a template df for shaping trades df
    template = pd.DataFrame(index=market_dates, columns = symbols).replace(np.NaN,1)

    #creating trades dataframe with only dates where the market was open
    trades = orders[orders.index.isin(market_dates)]

    #replacing BUY/SELL/HOLD with 1/-1/0 for easy computation
    trades.replace({'Order':{'BUY':1, 'SELL':-1, 'HOLD': 0}}, inplace=True)

    #creating trade column with the change in shares for each trade
    trades['Trade'] = trades['Order'] * trades['Shares']

    #creating a commissions and impact dfs to use later
    commissions_df = commission*(trades.groupby(['Date'])['Shares'].count() * template.iloc[:,0]).replace(np.NaN,0)
    impact_df = impact*(trades.groupby(['Date', 'Symbol'])['Shares'].sum().unstack(fill_value=0) * prices.iloc[:,:-1]).replace(np.NaN,0)

    #grouping trades df by date and symbol while summing the trades column
    trades = trades.groupby(['Date','Symbol'])['Trade'].sum().unstack(fill_value=0)
    trades.sort_index(inplace=True)  #sorting by order date

    #reordering columns and adding rows of zeros for market days without trades
    trades = trades[symbols]
    trades = (trades * template).replace(np.NaN,0)

    #adding cash column to trades df
    trades['Cash'] = (-1*trades * prices.loc[:,symbols]).replace(np.NaN,0).sum(axis=1)
    trades['Cash'] -= commissions_df.values
    trades['Cash'] -= impact_df.sum(axis=1)

    #adding initial cash value to first row of cash column
    trades.iloc[0,-1] += start_val

    #creating holdings, values, and  df
    holdings = trades.cumsum(axis=0)
    values = prices * holdings

    values = values[prices.columns]  #reordering values
    portvals = values.sum(axis=1)
    return portvals
