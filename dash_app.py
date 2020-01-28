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

app = dash.Dash(name=__name__)

periods_dict = {"5 days":"5d", "1 month":"1mo", "3 months":"3mo", 
                "6 months":"6mo", "1 year":"1y", "2 years":"2y", "5 years":"5y"}

#reading in NYSE stock tickers
tickers = pd.read_csv("yfinance_tickers.csv").iloc[:5,:]
tickers_str = ' '.join(tickers.Symbol.values)
#initializing data and graph
prices = pull_prices_viz(tickers_str, "5y")

#setting layout and title
app.title = "ML Stock Trader"
app.layout = html.Div(className='main-body', children=[
    #Top-left div
    html.Div(className='input-section', children=[
        #header
        dcc.Markdown(
            """
            # ML Stock Trader
            """
        ),
        
        #text box 2
        html.Br(),
        dcc.Dropdown(
            id='ticker',
            options=[{'label': i, 'value': i} for i in tickers.Symbol],
            value="AAPL",
            # style={'marmgin':'10px'}
        ),
    
        #text box 2
        html.Br(),
        dcc.Dropdown(
            id='timeframe',
            options=[{'label': i, 'value': i} for i in periods_dict.keys()],
            value="5 years",
            multi=False,
            placeholder="Time frame for plotting the stock price.",
            # style={'margin':'10px'}
        ),
        
        html.Br(),
        html.Div(id='company-name', children='Company: Apple, Inc.'),
        ],
    ),
    
    html.Br(),

    dcc.Graph(
            id='prices-plot',
            className="card-prices"
        )
    ],
)

@app.callback(
    [Output('prices-plot', 'figure'),
     Output('company-name', 'children')],
    [Input('ticker', 'value'),
     Input('timeframe', 'value')]
)
def create_plot(ticker, timeframe):
    #filtering prices by selected stock
    prices_one = prices.filter(items=["Date",ticker],axis=1)
    
    #splitting time input
    t_list = timeframe.split(' ')
    t_qty, t_unit = int(t_list[0]),t_list[1]

    #retrieving the start and end dates
    end_date =  datetime.datetime.today() #.today()
    if t_unit[:3] == "day":
        start_date = end_date - relativedelta(days=t_qty)
    elif t_unit[:5] == "month":
        start_date = end_date - relativedelta(months=t_qty)
    elif t_unit[:4] == "year":
        start_date = end_date - relativedelta(years=t_qty)
    
    #filtering prices by start and end dates
    mask = (prices_one['Date'] >= start_date) & (prices_one['Date'] <= end_date)
    prices_one = prices_one.loc[mask]
    
    #creating graph
    title = "{} Price over the last {}".format(ticker.upper(), "5 years")
    fig_new = px.line(prices_one, x="Date", y=ticker, title=title)
    
    #creating Company Name string
    company_name = "Company: {}".format(tickers[tickers.Symbol == ticker].Name.values[0])
    
    return fig_new, company_name



if __name__ == "__main__":
    app.run_server(debug=True)
