#importing dependencies
import numpy as np
import pandas as pd
import datetime as dt

import util as ut
from marketsimcode import *
from StrategyLearner import StrategyLearner
from ManualStrategy import ManualStrategy

def author():
    return 'rmiller327'

def test_code(learner, orders_sl_df, orders_ms_df, sd, ed, symbol, sv, plot_title, plot_indicators_bool = False):
    #reading in SPY prices over the in-sample date range
    dates = pd.date_range(sd,ed)
    prices = ut.get_data([symbol], dates).drop(['SPY'],axis=1)
    #test trades using marketsimcode
    portvals_sl = compute_portvals(orders_sl_df, start_val = sv, commission=0.0, impact=0.005)
    portvals_ms = compute_portvals(orders_ms_df, start_val = sv, commission=0.0, impact=0.005)

    prices['Manual Strategy'] = portvals_ms
    prices['Strategy Learner'] = portvals_sl

    plot_winnings(prices, plot_title, ["Manual Strategy"," Strategy Learner",
                  "Predicted Prices"], [], [])
    adr,sddr,cr = port_stats(portvals_sl)
    adr_ms,sddr_ms,cr_ms = port_stats(portvals_ms)

    #if plot_indicators_bool == True, then plotting the indicator graphs
    if plot_indicators_bool:
        price_sma_plot(prices, n = 10)
        bb_plot(prices, n = 10)
        volatility_plot(prices, n = 10)

    # Print statistics
    print(f"Start Date: {sd}")
    print(f"End Date: {ed}")
    print(f"Symbol: {symbol}")
    
    print(f"SL: Std Dev of Daily Returns: %.5f"%sddr)
    print(f"SL: Average Daily Return: %.5f"%adr)
    print(f"SL: Cumulative Return: %.5f"%cr)
    
    print(f"MS: Std Dev of Daily Returns: %.5f"%sddr_ms)
    print(f"MS: Average Daily Return: %.5f"%adr_ms)
    print(f"MS: Cumulative Return: %.5f"%cr_ms)


def experiment1():
    #initializing objects
    sl = StrategyLearner(impact=0.005)
    ms = ManualStrategy()
    
    #in-sample parameters
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    
    print()
    print("######################   In-Sample   #######################")
    print()
    
    #training Random Forest regressor
    sl.addEvidence(symbol, sd, ed, sv)
    
    #creating StrategyLearner orders dataframe
    trades_sl_df = sl.testPolicy(symbol, sd, ed, sv)
    orders_sl_df = sl.generate_orders_df(trades_sl_df, symbol)
    
    #creating ManualStrategy orders dataframe
    trades_ms_df = ms.testPolicy(symbol, sd, ed, sv)
    orders_ms_df = sl.generate_orders_df(trades_ms_df, symbol)
    
    #testing learner
    test_code(sl, orders_sl_df, orders_ms_df, sd, ed, symbol, sv,
              "Strategy Learner vs Manual Strategy (in-sample)", False)
    
    
    print()
    print("######################  Out-of-Sample  ######################")
    print()
    
    #out-of-sample dates
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)

    #creating StrategyLearner orders dataframe
    trades_sl_df = sl.testPolicy(symbol, sd, ed, sv)
    orders_sl_df = sl.generate_orders_df(trades_sl_df, symbol)
    
    #creating ManualStrategy orders dataframe
    trades_ms_df = ms.testPolicy(symbol, sd, ed, sv)
    orders_ms_df = sl.generate_orders_df(trades_ms_df, symbol)
    
    test_code(sl, orders_sl_df, orders_ms_df, sd, ed, symbol, sv,
              "Strategy Learner vs Manual Strategy (out-of-sample)", False)
              
              
if __name__ == "__main__":
    np.random.seed(12)
    experiment1()