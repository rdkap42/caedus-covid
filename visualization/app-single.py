import dash
from dash import Dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import datetime
from smart_open import open
import numpy as np
import pandas as pd
import os
import configparser 
from flask import Flask
import plotly.graph_objs as go
from plotly.offline import iplot
import gcsfs

""" Retrieve a configuration file for default information """
fileDir = os.getcwd()  
config = configparser.ConfigParser()
config.read(f'{fileDir}/config.ini')
default = config['default']
# Beds
coeff_bed = default.getfloat('default_coeff_bed')
coeff_low_bed = default.getfloat('default_coeff_low_bed')
coeff_up_bed = default.getfloat('default_coeff_up_bed')
# ICU Beds
coeff_icu = default.getfloat('default_coeff_icu')
coeff_low_icu = default.getfloat('default_coeff_low_icu')
coeff_up_icu = default.getfloat('default_coeff_up_icu')
# Vents
coeff_vents = default.getfloat('default_coeff_vents')
coeff_low_vents = default.getfloat('default_coeff_low_vents')
coeff_up_vents = default.getfloat('default_coeff_up_vents')
# Personnel
coeff_personnel = default.getfloat('default_coeff_personnel')
coeff_low_personnel = default.getfloat('default_coeff_low_personnel')
coeff_up_personnel = default.getfloat('default_coeff_up_personnel')

# Date calculations for forecasting
today = np.datetime64(datetime.date.today())
days5 = today + np.timedelta64(5, 'D')
days10 =  today + np.timedelta64(10, 'D')
days30 =  today + np.timedelta64(30, 'D')

# Read the static data source 
data_file = config['default']['default_datafile']
df_all = pd.read_csv(f'{data_file}')

df_all['Date']= pd.to_datetime(df_all['Date'])
#df_all['State']=df_all['State'].astype('string')
#df =  df_all[((df_all['Date']) < (today + np.timedelta64(150, 'D'))) & (df_all['State'] == 'MA')]
df = df_all[(df_all['Date']) < (today + np.timedelta64(150, 'D'))]
df = df[['Date', 'State','Cases_Mean', 'Cases_LB', 'Cases_UB', 'Deaths_Mean', 'Deaths_LB', 
'Deaths_UB', 'total_beds', 'total_ICU_beds', 'total_vents', 'phys_supply']]
df.columns = ['date','state','cases_mean', 'cases_lb', 'cases_ub', 'deaths_mean', 'deaths_lb', 
'deaths_ub', 'total_beds', 'total_ICU_beds', 'total_vents', 'phys_supply']

def generate_dcurve(dff):
    # ***Disease Curve Line Chart***
    lower_cases = go.Scatter(
        x=dff.date,
        y=dff.cases_lb,
        fill= None,
        mode='lines',
        showlegend = False,
        line=dict(
        color='#FDD663',
        width = 0
        )
    )
    upper_cases = go.Scatter(
        x=dff.date,
        y=dff.cases_ub,
        fill='tonexty',
        mode='lines',
        name = "95% CI cases",
        line=dict(
        color='#FDD663',
        width = 0
        )
    )
    mean_cases = go.Scatter(
        x=dff.date,
        y=dff.cases_mean,
        mode='lines+markers',
        name = "mean cases",
        line=dict(
            color='#FBBC04',
        )
    )
    lower_deaths = go.Scatter(
        x=dff.date,
        y=dff.deaths_lb,
        fill= None,
        mode='lines',
        showlegend = False,
        line=dict(
        color='#F28B82',
        width = 0
        )
    )
    upper_deaths = go.Scatter(
        x=dff.date,
        y=dff.deaths_ub,
        fill='tonexty',
        mode='lines',
        name = "95% CI deaths",
        line=dict(
        color='#F28B82',
        width = 0
        )
    )
    mean_deaths = go.Scatter(
        x=dff.date,
        y=dff.deaths_mean,
        mode='lines+markers',
        name = "mean deaths",
        line=dict(
            color='#EA4335',
        )
    )
    layout_disease = go.Layout(
    xaxis = dict(
        showline=True,
        showgrid=True,
    ),
    yaxis = dict(
        title = 'Estimated Total Cases',
        showline=True,
        showgrid=True,
    ),
    margin=go.layout.Margin(
            l=100, #left margin
            r=50, #right margin
            b=50, #bottom margin
            t=10, #top margin
        ),
    showlegend=True,
    transition={
        'duration': 1500,
        'easing': 'exp',
        'ordering': 'layout first'
    },
    legend=dict(
        orientation="h"
    ),
    annotations=[
        dict(
        x = today,
        y = dff[dff.date == today]["cases_mean"].values[0],
        xref = "x",
        yref="y",
        text = str(dff[dff.date == today]["cases_mean"].values[0]) + ' cases today',
        showarrow = True,
        arrowhead=7,
        ax = -30,
        ay = -30,
        font=dict(
            size=12
        ),
        )
    ]
    )
    data_disease = [lower_cases, upper_cases, mean_cases, lower_deaths, upper_deaths, mean_deaths]
    figure1 = dict(data=data_disease, layout =layout_disease)
    
    return dcc.Graph(
        id='graph',
        figure=figure1)

# App flask config
server = Flask(__name__)
server.debug = True
app = Dash(__name__, 
    server=server,
    external_stylesheets=[dbc.themes.LUX], 
    url_base_pathname='/')
app.config['suppress_callback_exceptions']=True

@server.route("/")
def MyDashApp():
    return app.index()

# Dash app layout
h1_viz_title = default.get('h1_viz_title')

# Configure navbar menu
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("About", href="#")),
        dbc.NavItem(dbc.NavLink("Test", href="#"))
    ],
    brand="Caedus Covid",
    brand_href="#",
    color="primary",
    dark=True,
    style={'text-align': 'left'}
)

app.layout = html.Div([
    html.Div([navbar]),
    html.Div([
      dcc.Dropdown(
        id='state-dropdown',
        options=[
            {'label': 'Alabama', 'value': '01'},
            {'label': 'Arizona', 'value': '02'},
            {'label': 'Arkansas', 'value': '03'},
            {'label': 'California', 'value': '04'},
            {'label': 'Colorado', 'value': '05'},
            {'label': 'Connecticut', 'value': '06'},
            {'label': 'Delaware', 'value': '07'},
            {'label': 'Florida', 'value': '08'},
            {'label': 'Georgia', 'value': '09'},
            {'label': 'Idaho', 'value': '10'},
            {'label': 'Illinios', 'value': '11'},
            {'label': 'Indiana', 'value': '12'},
            {'label': 'Iowa', 'value': '13'},
            {'label': 'Kansas', 'value': '14'},
            {'label': 'Kentucky', 'value': '15'},
            {'label': 'Louisiana', 'value': '16'},
            {'label': 'Maine', 'value': '17'},
            {'label': 'Maryland', 'value': '18'},
            {'label': 'Massachusetts', 'value': '19'},
            {'label': 'Michigan', 'value': '20'},
            {'label': 'Minnesota', 'value': '21'},
            {'label': 'Mississippi', 'value': '22'},
            {'label': 'Missouri', 'value': '23'},
            {'label': 'Montana', 'value': '24'},
            {'label': 'Nebraska', 'value': '25'},
            {'label': 'Nevada', 'value': '26'},
            {'label': 'New Hampshire', 'value': '27'},
            {'label': 'New Jersey', 'value': '28'},
            {'label': 'New Mexico', 'value': '29'},
            {'label': 'New York', 'value': '30'},
            {'label': 'North Carolina', 'value': '31'},
            {'label': 'North Dakota', 'value': '32'},
            {'label': 'Ohio', 'value': '33'},
            {'label': 'Oklahoma', 'value': '34'},
            {'label': 'Oregon', 'value': '35'},
            {'label': 'Pennsylvania', 'value': '36'},
            {'label': 'Rhode Island', 'value': '37'},
            {'label': 'South Carolina', 'value': '38'},
            {'label': 'South Dakota', 'value': '29'},
            {'label': 'Tennessee', 'value': '40'},
            {'label': 'Texas', 'value': '41'},
            {'label': 'Utah', 'value': '42'},
            {'label': 'Vermont', 'value': '43'},
            {'label': 'Virginia', 'value': '44'},
            {'label': 'Washington', 'value': '45'},
            {'label': 'West Virginia', 'value': '46'},
            {'label': 'Wisconsin', 'value': '47'},
            {'label': 'Wyoming', 'value': '48'}
    ],
    style={'marginBottom': '30px', 'marginTop': '10px', 
    'text-align': 'center',
    'text-transform': 'uppercase',
    'font-size':'100%',
    'height': '30px'},
    placeholder="Select a state",
    value=['19'], # default value
    )]),
    html.H4('Number of Cases: Statewide',style={'marginLeft': '20px'}),
    html.Div(id='line-chart1', children=[])
    ])
@app.callback(
    dash.dependencies.Output('line-chart1', 'children'),
    [dash.dependencies.Input('state-dropdown', 'value')])
def display_dcurve(value):
      
    #if value is None:
    #    return generate_dcurve(df)
        
    dff = df[df.state.isin([value])]
    print(dff)
    return generate_dcurve(dff)

if __name__ == '__main__':
    app.run_server( host='0.0.0.0', port=os.environ.get('PORT',8050), dev_tools_hot_reload=False)