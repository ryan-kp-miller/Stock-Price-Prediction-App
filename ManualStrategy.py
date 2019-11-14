"""
Student Name: Ryan Miller
GT User ID: rmiller327
GT ID: 903461824
"""

#importing dependencies
from indicators import *
from marketsimcode import *
import pandas as pd
import datetime as dt
from util import get_data

class ManualStrategy:
    """
        class implementing a manual set of rules using the indicators
        in indicators.py to enter/exit/hold positions in each given stock
    """
    def __init__(self):
        pass

    def author(self):
        return 'rmiller327'


    def testPolicy(self, symbol, sd, ed, sv):
        """
            method that uses the indicators in indicators.py to decide when to
            buy, sell, or hold for a given stock over the trading days between
            the given start and end dates with the given starting value

            inputs:
                symbol:    string representing the stock symbol to trade
                sd:        datetime object representing the date to start trading
                ed:        datetime object representing the date to stop trading
                sv:        integer representing the starting amount of money you
                           have to trade with
            output:
                df_trades: pandas dataframe containing a trade (int) for each
                           trading day between sd and ed
        """
        #reading in the stock data using util.py and removing nulls
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)
        prices.drop(['SPY'], axis=1, inplace=True) #dropping SPY column
        prices.fillna(method='ffill', inplace=True) #forward-filling missing prices
        prices.fillna(method='bfill', inplace=True) #back-filling missing prices

        #initializing df_trades to all 0s
        df_trades = prices.copy()
        df_trades.iloc[:,0] = 0

        #creating the indicator dfs
        n = 9
        price_sma_df = price_sma_ratio(prices, n)
        bb_df = bollinger_bands(prices, n)
        vol_df = volatility(prices, n)

        #looping through the trading days and buying/selling/holding based on
        #the indicators using the stock price; keeping counter variable for
        #current holdings; current_holdings cannot exceed +/- 1000
        current_holdings = 0
        for i in range(df_trades.shape[0]-1):
            #at least one of these conditions need to be met to signal a buy
            buy_bool = bb_df.iloc[i,0] < -1.5 or price_sma_df.iloc[i,0] < -1.8 or vol_df.iloc[i,0] > 1.2
            #at least one of these conditions need to be met to signal a buy
            sell_bool = bb_df.iloc[i,0] > 1.3 or price_sma_df.iloc[i,0] > 1.65 or vol_df.iloc[i,0] < -0.6
            #buying/selling if bb exceeds +/- 1, else holding
            if buy_bool:  #buying conditions are met
                df_trades.iloc[i,0] = 1000 - current_holdings #buy as much as we can
                current_holdings = 1000  #setting to current holdings
            elif sell_bool:  #price goes down
                df_trades.iloc[i,0] = -1000 - current_holdings #sell as much as we can
                current_holdings = -1000 #setting to current holdings
            #if neither buy_bool or sell_bool are true, then that day's trade = 0
        return df_trades


def test_code(trades, sd, ed, symbol, plot_title, plot_indicators_bool = False):
    #creating orders df to use old marketsim compute_portvals function
    orders = trades.copy()
    orders['Order'] = ['BUY' if i > 0 else 'SELL' for i in orders.iloc[:,0].values]
    orders['Date'] = orders.index.values
    orders['Shares'] = abs(orders.iloc[:,0].values)
    orders['Symbol'] = symbol
    orders.drop([symbol],axis=1,inplace=True)
    #creating long and short lists
    long_list = [orders.index[i] for i in range(orders.shape[0]) if orders.Order[i] == 'BUY']
    short_list = [orders.index[i] for i in range(orders.shape[0]) if orders.Order[i] == 'SELL' and orders.Shares[i] > 0]

    #reading in SPY prices over the in-sample date range
    dates = pd.date_range(sd,ed)
    prices = get_data([symbol], dates).drop(['SPY'],axis=1)
    #test trades using marketsimcode
    portvals = compute_portvals(orders, start_val = sv, commission=9.95, impact=0.005)
    prices['MS'] = portvals

    plot_winnings(prices, plot_title, ["Benchmark","Manual Strategy"],
                  long_list, short_list)
    adr,sddr,cr = port_stats(portvals)
    jpm_adr,jpm_sddr,jpm_cr = port_stats(prices[symbol])

    #if plot_indicators_bool == True, then plotting the indicator graphs
    if plot_indicators_bool:
        price_sma_plot(prices, n = 10)
        bb_plot(prices, n = 10)
        volatility_plot(prices, n = 10)


    # Print statistics
    print(f"Start Date: {sd}")
    print(f"End Date: {ed}")
    print(f"Symbol: {symbol}")
    print(f"MS: Std Dev of Daily Returns: {sddr}")
    print(f"MS: Average Daily Return: {adr}")
    print(f"MS: Cumulative Return: {cr}")
    print(f"Benchmark: Std Dev of Daily Returns: {jpm_sddr}")
    print(f"Benchmark: Average Daily Return: {jpm_adr}")
    print(f"Benchmark: Cumulative Return: {jpm_cr}")


if __name__ == "__main__":
    ms = ManualStrategy()
    #in-sample parameters
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    df_trades = ms.TestPolicy(symbol, sd, ed, sv)
    test_code(df_trades, sd, ed, symbol,
              "Manual Strategy vs Benchmark (in-sample)", True)

    #out-of-sample parameters
    symbol = "JPM"
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000
    df_trades = ms.TestPolicy(symbol, sd, ed, sv)
    test_code(df_trades, sd, ed, symbol,
              "Manual Strategy vs Benchmark (out-of-sample)", False)
