import pandas as pd
import numpy as np
import yfinance as yf
from indicators import *
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
from joblib import dump,load
import datetime as dt
import os

class MLTrader:
    """
        class for making daily predictions of a given stock's price

        inputs:
            learner: ML object to be trained and used for prediction
                     must have fit(X,Y) and predict(X) methods
            impact:  float representing the assumed (and simplified) impact that
                     a trade will have on the price of the stock
            n:       integer representing the length of the rollling windows for
                     generating the indictors
            kwargs:  dictionary containing the arguments to feed into the ML
                     learner object
    """

    def __init__(self, learner, impact = 0.0, n = 9, kwargs = {}):
        self.learner = None if learner is None else learner(**kwargs)
        self.impact = impact
        self.n = n
        #storing sklearn's standard scaler for scaling the train and test prices
        self.ss = StandardScaler()
        #counter variables to track the total number of total and bad trades
        self.trades = 0
        self.bad_trades = 0


    def preprocess_data(self, symbol, sd, ed):
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
        #creating a list with the input symbol and "SPY" to pull in all days
        #the stock market was open
        symbols = [symbol, "SPY"]

        #reading in the stock data using util.py and removing nulls
        df = yf.download(symbols, start=sd, end=ed, auto_adjust=True)
        prices = df['Close'].filter(items=[symbol], axis=1)

        prices.fillna(method='ffill', inplace=True) #forward-filling missing prices
        prices.fillna(method='bfill', inplace=True) #back-filling missing prices
        return prices


    def generate_indicators(self, prices):
        """
            helper method for generating features dataframe containing the
            training indicators

            input:
                prices:        dataframe containing the daily prices of a stock

            output:
                indicators_df: dataframe containing the Price/SMA Ratio,
                               Bollinger Bands, and Volatility of the daily
                               stock prices
        """
        #creating the indicator dfs
        price_sma_df = price_sma_ratio(prices, self.n)
        bb_df = bollinger_bands(prices, self.n)
        vol_df = volatility(prices, self.n)

        #creating features dataframe for training the regressor
        indicators_df = price_sma_df.join(bb_df,rsuffix="1") \
                                    .join(vol_df,rsuffix="2")
        indicators_df.columns = ["Price/SMA", "Bollinger Bands", "Volatility"]

        #adding column for the closing price of one days prior
        indicators_df["Previous Price"] = prices.shift(1).values

        return indicators_df


    def generate_orders_df(self, trades_df, symbol):
        """
            helper method for creating the order dataframe needed to run the
            market simulator

            input:
                trades_df: dataframe containing a trade (int) for each
                           trading day between sd and ed

            output:
                orders_df: dataframe containing the order type, date, number of
                           shares traded, and the stock symbol being traded for
                           each trade in trades_df
        """
        orders_df = trades_df.copy()
        orders_df.loc[orders_df['Trade'] > 0 ,'Order'] = 'BUY'
        orders_df.loc[orders_df['Trade'] < 0 ,'Order'] = 'SELL'
        orders_df.loc[orders_df['Trade'] == 0 ,'Order'] = 'HOLD'
        # orders_df['Date'] = orders_df.index.values
        orders_df['Shares'] = abs(orders_df.iloc[:,0].values)
        orders_df['Symbol'] = symbol
        orders_df.drop(['Trade'],axis=1,inplace=True)
        return orders_df


    def fit(self, symbol = "IBM", sd = "2008-01-01", ed = "2009-01-01"):
        """
            method for training the given ML regressor object for predicting the
            normalized price of the given stock

            inputs:
                symbol: string representing the stock symbol to trade
                sd:     string representing the date to start trading
                ed:     string representing the date to stop trading

            output:
                None
        """
        #reading in the price data
        prices = self.preprocess_data(symbol, sd, ed)
        #normalizing the prices
        prices_norm = pd.DataFrame(self.ss.fit_transform(prices),index=prices.index, columns=[symbol])
        #generating indicator dataframe for predicting
        features_df = self.generate_indicators(prices_norm).fillna(method='bfill')
        #training regressor to predict prices using the indicators in indicators.py
        self.learner.fit(features_df.values, prices_norm.values)


    def save_learner(self, symbol = ""):
        """
            method that saves the learner using joblib
            assumes that the model is from scikit-learn
        """
        #creating models folder if it doesn't already exist
        folder_path = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(folder_path):
             os.mkdir(folder_path)

        #dumping the ML model and StandardScaler objects into the models folder
        dump(self.learner, os.path.join(folder_path, "{}_model.joblib".format(symbol)))
        dump(self.ss, os.path.join(folder_path, "{}_ss.joblib".format(symbol)))


    def load_learner(self, symbol = ""):
        """
            method that loads the learner using joblib
            assumes that the model was saved using save_learner method
        """
        self.learner = load("models/{}_model.joblib".format(symbol))
        self.ss = load("models/{}_ss.joblib".format(symbol))


    def predict_today(self, symbol):
        """
            method to predict the adjusted closing stock price for today

            input:
                symbol: string representing the stock symbol for trading

            output:
                price:  float representing today's predicted closing stock price
                        for the given symbol
        """
        #finding the start_date based on the 2*window-length to account for
        #days the market isn't open
        ed = dt.datetime.today() - relativedelta(days=1)
        sd = ed - relativedelta(days=self.n*2)

        #reading in the prices data and normalizing it
        prices = self.preprocess_data(symbol, sd, ed)
        prices_norm = pd.DataFrame(self.ss.transform(prices),index=prices.index,
                                   columns=[symbol])

        #generating indicator dataframe for predicting
        features_df = self.generate_indicators(prices_norm)

        #removing the n days of blanks from prices and features_df
        prices_norm = prices_norm.iloc[self.n:,:]
        features_df = features_df.iloc[self.n:,:]

        #predicting prices using Random Forest regressor
        prices_array = self.learner.predict(features_df.values)
        price = self.ss.inverse_transform(prices_array)[0,0]
        return price


    def testLearner(self, symbol = "IBM", sd = "2009-01-01", ed = "2010-01-01"):
        """
            method for using the trained ML regressor to create a dataframe of
            stock trades for the given time period and stock to be fed into the
            market simulator

            inputs:
                symbol:    string representing the stock symbol to trade
                sd:        string representing the date to start trading
                ed:        string representing the date to stop trading

            output:
                df_trades: pandas dataframe containing a trade (int) for each
                           trading day between sd and ed
        """
        #reading in the price data and normalizing it
        prices = self.preprocess_data(symbol, sd, ed)
        prices_norm = pd.DataFrame(self.ss.transform(prices),index=prices.index,
                                   columns=[symbol])

        #generating indicator dataframe for predicting
        features_df = self.generate_indicators(prices_norm)

        #removing the n days of blanks from prices and features_df
        prices_norm = prices_norm.iloc[self.n:,:]
        features_df = features_df.iloc[self.n:,:]

        #predicting prices using Random Forest regressor
        prices_array = self.learner.predict(features_df.values)
        self.prices_pred = pd.DataFrame(prices_array,  columns=[symbol],
                                        index=prices_norm.index.values)

        #initializing df_trades to all 0s
        trades_df = pd.DataFrame(np.zeros((prices_array.shape[0], 1)),
                                 index=self.prices_pred.index,
                                 columns = ['Trade'])
        trades_df.index.rename('Date', inplace=True)

        #looping through the trading days and buying/selling/holding based on
        #the next day's predicted stock price; keeping counter variable for
        #current holdings (current_holdings cannot exceed +/- 1000)
        current_holdings = 0
        #resetting total trades and bad trades counter variables
        self.trades = 0
        self.bad_trades = 0
        #starting at trading day n because we cannot trade using backfilled data
        for i in range(self.n,trades_df.shape[0]-1):
            #buying/selling if price goes up/down, else holding
            if self.prices_pred.iloc[i+1,0] > prices_norm.iloc[i,0]*(1+self.impact):  #price goes up
                trades_df.iloc[i,0] = 1000 - current_holdings #buy as much as we can
                current_holdings = 1000 #setting to current holdings
                #counting the total number of trades for later comparison
                self.trades+=1
                #counting the number of times the learner shouldn't have traded
                if prices_norm.iloc[i+1,0] < prices_norm.iloc[i,0]:
                    self.bad_trades+=1
            elif self.prices_pred.iloc[i+1,0] < prices_norm.iloc[i,0]*(1-self.impact):  #price goes down
                trades_df.iloc[i,0] = -1000 - current_holdings #sell as much as we can
                current_holdings = -1000  #setting to current
                #counting the total number of trades for later comparison
                self.trades+=1
                #counting the number of times the learner shouldn't have traded
                if prices_norm.iloc[i+1,0] > prices_norm.iloc[i,0]:
                    self.bad_trades+=1
        return trades_df
