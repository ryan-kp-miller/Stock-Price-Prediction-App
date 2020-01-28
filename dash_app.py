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

periods_list = ["5 Days", "1 Month", "3 Months", "6 Months", 
                "1 Year", "2 Years", "5 Years"]

#reading in NYSE stock tickers
tickers = pd.read_csv("yfinance_tickers.csv").iloc[:5,:]
tickers_str = ' '.join(tickers.Symbol.values)
#initializing data and graph
prices = pull_prices_viz(tickers_str, "5y")

#setting layout and title
app.title = "ML Stock Trader"
app.layout = html.Div(className='main-body', children=[
    #Top-left div
    html.Div(id='card-outer', className='card-outer', children=[
        html.Div(id='card-1', className='card', children=[
            html.Br(),
            #header
            html.H3(className='header',children="ML Stock Trader"),
            
            #text box 2
            html.Br(),
            dcc.Dropdown(
                id='ticker',
                options=[{'label': i, 'value': i} for i in tickers.Symbol],
                value="AAPL",
            ),
        
            #text box 2
            html.Br(),
            dcc.Dropdown(
                id='timeframe',
                options=[{'label': i, 'value': i} for i in periods_list],
                value="5 Years",
                multi=False,
            ),
            
            html.Br(),
            html.Div(id='company-name', children='Company: Apple, Inc.'),
            html.Br(),
            ],
        ),
    
        html.Div(id='card-2', className='card', children=[
            html.H3(className='header', children="Stock Price Prediction Card")    
        ]),
        html.Div(id='card-3', className='card', children=[
            html.H3(className='header', children="Sentiment Analysis Card") 
        ]),
    ]),
    
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
    fig["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C",
                         title={'xanchor':'center', 'y':0.9, 'x':0.5,
                                'font':{'color':'white'}},
                         xaxis={'showgrid': False, 'color':'white'},
                         yaxis={'showgrid': False, 'color':'white'})
    
    #creating Company Name string
    company_name = "Company: {}".format(tickers[tickers.Symbol == ticker].Name.values[0])
    
    return fig, company_name



if __name__ == "__main__":
    app.run_server(debug=True)
