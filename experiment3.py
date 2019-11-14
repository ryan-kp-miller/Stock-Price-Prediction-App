#importing dependencies
import yfinance as yf
import pandas as pd

from StrategyLearner import StrategyLearner
from marketsimcode import *


def test_code(learner, orders_sl_df, sd, ed, symbol, sv, plot_title, plot_indicators_bool = False):
    #reading in SPY prices over the in-sample date range
    prices = learner.preprocess_data(symbol, sd, ed)
    #test trades using marketsimcode
    portvals_sl = compute_portvals(orders_sl_df, start_val = sv, commission=0.0, impact=0.005)

    prices['Strategy Learner'] = portvals_sl

    plot_winnings(prices, plot_title, ["Strategy Learner",
                  "Prices"], [], [])
    adr,sddr,cr = port_stats(portvals_sl)

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


def experiment3():
    #initializing objects
    sl = StrategyLearner(impact=0.005)
    
    #in-sample parameters
    symbol = "JPM"
    sd = "2008-01-01"
    ed = "2009-12-31"

    sv = 100000
    
    print()
    print("######################   In-Sample   #######################")
    print()
    
    #training Random Forest regressor
    sl.addEvidence(symbol, sd, ed, sv)
    
    #creating StrategyLearner orders dataframe
    trades_sl_df = sl.testPolicy(symbol, sd, ed, sv)
    orders_sl_df = sl.generate_orders_df(trades_sl_df, symbol)
    
    #testing learner
    test_code(sl, orders_sl_df, sd, ed, symbol, sv,
              "Strategy Learner vs Stock Price (in-sample)", False)
    
    
    print()
    print("######################  Out-of-Sample  ######################")
    print()
    
    #out-of-sample dates
    sd = "2010-01-01"
    ed = "2012-12-31"

    #creating StrategyLearner orders dataframe
    trades_sl_df = sl.testPolicy(symbol, sd, ed, sv)
    orders_sl_df = sl.generate_orders_df(trades_sl_df, symbol)
        
    test_code(sl, orders_sl_df, sd, ed, symbol, sv,
              "Strategy Learner vs Stock Price (out-of-sample)", False)
              
              
if __name__ == "__main__":
    np.random.seed(12)
    experiment3()