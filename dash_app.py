import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px

import pandas as pd
import datetime as dt
import yfinance as yf
from MLTrader import MLTrader
from market_simulator import pull_prices_viz
from dateutil.relativedelta import relativedelta

app = dash.Dash(name=__name__)

periods_list = ["5 Days", "1 Month", "3 Months", "6 Months",
                "1 Year", "2 Years", "5 Years"]

#reading in NYSE stock tickers
tickers = pd.read_csv("yfinance_tickers.csv")
tickers_str = ' '.join(tickers.Symbol.values)
#initializing data and graph
prices = pull_prices_viz(tickers_str, "5y")

#setting layout and title
app.title = "ML Stock Trader"
app.layout = html.Div(className='main-body', children=[
    #Top-left div
    html.Div(id='card-outer', className='card-outer', children=[
        html.Div(id='card-1', className='card-input', children=[
            #header
            html.H3(className='header',children="ML Stock Trader"),

            #text box 2
            dcc.Dropdown(
                id='company-name',
                className='dropdown',
                options=[{'label': i, 'value': i} for i in tickers.Name],
                value=tickers.Name[0],
            ),

            #text box 2
            html.Br(),
            dcc.Dropdown(
                id='timeframe',
                className='dropdown',
                options=[{'label': i, 'value': i} for i in periods_list],
                value="5 Years",
                multi=False,
            )],
        ),

        html.Div(id='card-2', className='card', children=[
            html.H3(className='header', children="Stock Price Prediction"),
            html.Div(id='company-ticker', children='Ticker: AAPL',
                     style={'textAlign':'center'}),
            html.Br(),
            html.Div(id="current-price",style={'color':'white',
                                               'textAlign':'center'}),
            html.Br(),
            html.Div(id="predicted-price"),
        ]),
        html.Div(id='card-3', className='card', children=[
            html.H3(className='header', children="Sentiment Analysis"),
        ]),
    ]),

    html.Br(),

    html.Div(id='prices-div', style={'height':'50vh'},
             children=dcc.Graph(id='prices-plot', style={'height':'inherit'},
                                config={'responsive':True})),
    ],
)

@app.callback(
    Output('prices-plot', 'figure'),
    [Input('company-name', 'value'),
     Input('timeframe', 'value')]
)
def create_plot(name, timeframe):
    #retrieving stock ticker
    ticker = tickers[tickers.Name == name].Symbol.values[0]

    #filtering prices by selected stock
    prices_one = prices.filter(items=["Date",ticker],axis=1)

    #splitting time input
    t_list = timeframe.split(' ')
    t_qty, t_unit = int(t_list[0]),t_list[1]

    #retrieving the start and end dates
    end_date =  dt.datetime.today() #.today()
    if t_unit[:3] == "Day":
        start_date = end_date - relativedelta(days=t_qty)
    elif t_unit[:5] == "Month":
        start_date = end_date - relativedelta(months=t_qty)
    else:
        start_date = end_date - relativedelta(years=t_qty)

    #filtering prices by start and end dates
    mask = (prices_one['Date'] >= start_date) & (prices_one['Date'] <= end_date)
    prices_one = prices_one.loc[mask]

    #creating graph
    title = "{} Price over the last {}".format(ticker.upper(), timeframe)
    fig = px.line(prices_one, x="Date", y=ticker, title=title)

    #updating graph layout (docs: https://plot.ly/python/reference/#layout)
    fig["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C",
                         title={'xanchor':'center', 'y':0.9, 'x':0.5,
                                'font':{'color':'white'}},
                         xaxis={'showgrid': False, 'color':'white'},
                         yaxis={'showgrid': False, 'color':'white',
                                'title':'Stock Price'},
                         height=350)
    return fig


@app.callback(
    [Output('current-price', 'children'),
     Output('predicted-price', 'children'),
     Output('predicted-price', 'style'),
     Output('company-ticker', 'children')],
    [Input('company-name', 'value')]
)
def show_prices(name):
    #retrieving stock ticker
    ticker = tickers[tickers.Name == name].Symbol.values[0]

    #creating the trader and loading the given stock's model
    trader = MLTrader(None, n=10)
    trader.load_learner(ticker)

    #getting the current stock price and predicting tomorrow's price
    current_price = round(prices[ticker].values[-1],2)
    predicted_price = round(trader.predict_today(ticker),2)

    #deciding if the predicted price is higher or lower than the current price
    if predicted_price > current_price:
        color = "green"
    elif predicted_price < current_price:
        color = "red"
    else:
        color = "white"
        
    #formatting strings to display
    current_str = "Yesterday's Closing Price: ${:,.2f}".format(current_price)
    predicted_str = "Today's Predicted Closing Price: ${:,.2f}".format(predicted_price)
    predicted_style = {'color':color, 'textAlign':'center'}
    ticker_str = "Stock Ticker: \t{}".format(ticker)
    return current_str,predicted_str,predicted_style,ticker_str


if __name__ == "__main__":
    app.run_server(debug=True)
