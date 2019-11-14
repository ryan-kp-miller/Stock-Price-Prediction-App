#importing dependencies
import numpy as np
import pandas as pd
import datetime as dt

import util as ut
from StrategyLearner import *
from marketsimcode import *

def author():
    return 'rmiller327'

def experiment2():
    #in-sample parameters
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    dates = pd.date_range(sd,ed)
    sv = 100000   
        
    #reading in SPY prices over the in-sample date range
    prices = ut.get_data([symbol], dates).drop(['SPY'],axis=1)
    #initializing object
    sl = StrategyLearner()    
    #training Random Forest regressor
    sl.addEvidence(symbol, sd, ed, sv)
    
    impact_list = [0, 0.005, 0.01]
    for i in range(len(impact_list)):
        #setting impact attribute for StrategyLearner object
        sl.impact = impact_list[i]
        #creating orders dataframe
        trades_df = sl.testPolicy(symbol, sd, ed, sv)
        orders_df = sl.generate_orders_df(trades_df, symbol)
        #test trades using marketsimcode
        portvals = compute_portvals(orders_df, start_val = sv, commission=0.0, impact=impact_list[i])
        adr,sddr,cr = port_stats(portvals)
        #counting number of nonzero trades
        num_trades_array = trades_df.astype(bool).sum() 

        #printing performance statistics for current impact value
        print("Impact: %.3f"%impact_list[i])
        print("Cumulative Return: %.5f"%cr)
        print("Number of Trades: %.5f"%num_trades_array)
        print()
        
if __name__ == "__main__":
    np.random.seed(12)
    experiment2()