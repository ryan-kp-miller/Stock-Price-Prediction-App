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
from market_simulator import pull_prices

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#initializing data and graph
end_date = datetime.datetime.today().date()
start_date = (end_date - relativedelta(days=5))
prices = pull_prices("AAPL",start_date, end_date).reset_index()

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

    #text box 1
    dcc.Input(
        id="ticker",
        type="text",
        # placeholder="AAPL",
        value="AAPL",
    ),

    html.Br(),

    #text box 2
    dcc.Dropdown(
        id='timeframe',
        options=[{'label': i, 'value': i} for i in ['5 days', '1 month', '1 year', '5 years']],
        value="5 days",
        multi=False,
        placeholder="Time frame for plotting the stock price.",
        style={'width':'50%'}),

    # dcc.Dropdown()

    html.Br(),
    html.Br(),
    #puppy image to make things more visually pleasing
    html.Div(id='start-date'),
    html.Div(id='end-date'),

    html.Br(),
    html.Br(),

    dcc.Graph(
            id='prices-plot',
            figure=fig
        )
])

@app.callback(
    [Output('start-date', 'children'),
     Output('end-date', 'children'),
     Output('prices-plot','figure')],
    [Input('ticker','value'),
     Input('timeframe','value')]
)
def create_plot(ticker, timeframe):
    #splitting time input
    time_list = timeframe.split()
    time_qty = int(time_list[0])
    time_unit = time_list[1]

    #retrieving todays date
    end_date = t = datetime.datetime.today() #.today()
    if time_unit[:3] == "day":
        start_date = end_date - relativedelta(days=time_qty)
    elif time_unit[:5] == "month":
        start_date = end_date - relativedelta(months=time_qty)
    elif time_unit[:4] == "year":
        start_date = end_date - relativedelta(years=time_qty)
    prices = pull_prices(ticker, start_date.date(), end_date.date()).reset_index()
    title = "{} Price over the last {}".format(ticker,timeframe)
    fig = px.line(prices, x="Date", y=ticker,
                  title=title)
    return "Start date: {}".format(start_date.date()),"End date: {}".format(end_date.date()),fig

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
