import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px

from MLTrader import MLTrader
import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from market_simulator import pull_prices_viz

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

periods_dict = {"5 days":"5d", "1 month":"1mo", "3 months":"3mo", 
                "6 months":"6mo", "1 year":"1y", "2 years":"2y", "5 years":"5y",
                "10 years":"10y", "year to date":"ytd"}

#reading in NYSE stock tickers
tickers = pd.read_csv("yfinance_tickers.csv")

#initializing data and graph
prices = pull_prices_viz("AAPL", "5d")
fig = px.line(prices, x="Date", y="AAPL",
              title="AAPL Price over the last 5 days")


#setting layout and title
app.title = "ML Stock Trader"
app.layout = html.Div([
    #header
    dcc.Markdown(
    """
    # ML Stock Trader
    """
    ),

    html.Br(),
    
    #text box 2
    dcc.Dropdown(
        id='ticker',
        options=[{'label': i, 'value': i} for i in tickers.Symbol],
        value="AAPL",
        style={'width':'50%'}),

    html.Br(),

    #text box 2
    dcc.Dropdown(
        id='timeframe',
        options=[{'label': i, 'value': i} for i in periods_dict.keys()],
        value="5 days",
        multi=False,
        placeholder="Time frame for plotting the stock price.",
        style={'width':'50%'}),

    # dcc.Dropdown()
    html.Br(),
    html.Br(),
    html.Div(id='error-log'),
    

    html.Br(),
    
    html.Div(id='company-name', children='Company: Apple, Inc.'),
    
    html.Br(),

    dcc.Graph(
            id='prices-plot',
            figure=fig
        )
])

@app.callback(
    [Output('prices-plot', 'figure'),
     Output('company-name', 'children'),
     Output('error-log',   'children')],
    [Input('ticker',       'value'),
     Input('timeframe',    'value')],
    [State('prices-plot',  'figure')]
)
def create_plot(ticker, timeframe, fig_old):
    #retrieving the stock prices
    period = periods_dict[timeframe]
    prices = pull_prices_viz(ticker.upper(), period)
    #only updating the figure if the prices were pulled correctly
    if prices.shape[0] > 0:
        title = "{} Price over the last {}".format(ticker.upper(), timeframe)
        fig_new = px.line(prices, x="Date", y=ticker, title=title)
        company_name = "Company: {}".format(tickers[tickers.Symbol == ticker].Name.values[0])
        return fig_new, company_name, ""

# @app.callback(
#     [Output('prices-plot', 'figure'),
#      Output('error-log',   'children')],
#     [Input('ticker',       'value'),
#      Input('timeframe',    'value')],
#     [State('prices-plot',  'figure')]
# )
# def create_plot(ticker, timeframe, fig_old):
#     #only attempting to pull data if the stock ticker is over 2 characters long
#     if len(ticker) <= 2 :
#         return fig_old, "Incorrect stock ticker"
#     else:  
#         try:  
#             #retrieving the stock prices
#             period = periods_dict[timeframe]
#             prices = pull_prices_viz(ticker.upper(), period)
#             #only updating the figure if the prices were pulled correctly
#             if prices.shape[0] > 0:
#                 title = "{} Price over the last {}".format(ticker.upper(), timeframe)
#                 fig_new = px.line(prices, x="Date", y=ticker, title=title)
#                 return fig_new, ""
#         except:
#             return fig_old, "Incorrect stock ticker"

    #toggles puppy-image
    # @app.callback(
    #     Output('puppy-display', 'children'),
    #     [Input('puppy-button', 'n_clicks')])
    # def toggle_puppy(n_clicks):
    #     if n_clicks % 2 != 0:
    #         prices = pull_prices(ticker, start_date.date(), end_date.date()).reset_index()
    #         title = "{} Price over the last {}".format(ticker,timeframe)
    #         fig = px.line(prices, x="Date", y=ticker,
    #                       title=title)
            # return fig


if __name__ == "__main__":
    app.run_server(debug=True)
