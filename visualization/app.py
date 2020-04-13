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

""" Retrieve a configuration file for default information """
fileDir = os.getcwd()  
config = configparser.ConfigParser()
config.read(f'{fileDir}/config.ini')

default = config['default']
data_file = config['default']['data_file_path']
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

h1_viz_title = default.get('h1_viz_title')

# Date calculations for forecasting
today = np.datetime64(datetime.date.today())
days5 = today + np.timedelta64(5, 'D')
days10 =  today + np.timedelta64(10, 'D')
days30 =  today + np.timedelta64(30, 'D')

def parse_data():
    # Read the static data source 
    df_all = pd.read_csv(open(f'{data_file}'))
    today = np.datetime64(datetime.date.today())

    df_all['Date']= pd.to_datetime(df_all['Date'])
    print(df_all)
    df =  df_all[(df_all['Date']) < (today + np.timedelta64(150, 'D'))]
    df = df[['Date','Cases_Mean', 'Cases_LB', 'Cases_UB', 'Deaths_Mean', 'Deaths_LB', 
    'Deaths_UB', 'total_beds', 'total_ICU_beds', 'total_vents', 'phys_supply']]
    df.columns = ['date','cases_mean', 'cases_lb', 'cases_ub', 'deaths_mean', 'deaths_lb', 
    'deaths_ub', 'total_beds', 'total_ICU_beds', 'total_vents', 'phys_supply']

    return df

df = parse_data()

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

# ***Disease Curve Line Chart***
lower_cases = go.Scatter(
    x=df.date,
    y=df.cases_lb,
    fill= None,
    mode='lines',
    showlegend = False,
    line=dict(
      color='#FDD663',
      width = 0
    )
)
upper_cases = go.Scatter(
    x=df.date,
    y=df.cases_ub,
    fill='tonexty',
    mode='lines',
    name = "95% CI cases",
    line=dict(
      color='#FDD663',
      width = 0
    )
)
mean_cases = go.Scatter(
    x=df.date,
    y=df.cases_mean,
    mode='lines+markers',
    name = "mean cases",
    line=dict(
        color='#FBBC04',
    )
)
lower_deaths = go.Scatter(
    x=df.date,
    y=df.deaths_lb,
    fill= None,
    mode='lines',
    showlegend = False,
    line=dict(
      color='#F28B82',
      width = 0
    )
)
upper_deaths = go.Scatter(
    x=df.date,
    y=df.deaths_ub,
    fill='tonexty',
    mode='lines',
    name = "95% CI deaths",
    line=dict(
      color='#F28B82',
      width = 0
    )
)
mean_deaths = go.Scatter(
    x=df.date,
    y=df.deaths_mean,
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
  legend=dict(
    orientation="h"
  ),
  annotations=[
    dict(
      x = today,
      y = df[df.date == today]["cases_mean"].values[0],
      xref = "x",
      yref="y",
      text = str(df[df.date == today]["cases_mean"].values[0]) + ' cases today',
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

# ***Hospital Bed Bar Chart***
today_val = df[df.date == today]["cases_mean"].values[0] * coeff_bed
day5_val = df[df.date == days5]["cases_mean"].values[0] * coeff_bed
day10_val= df[df.date == days10]["cases_mean"].values[0] * coeff_bed
day30_val = df[df.date == days30]["cases_mean"].values[0] * coeff_bed
capacity = df[df.date == today]["total_beds"].values[0]

# For Error Bars
today_up = (df[df.date == today]["cases_ub"].values[0] * coeff_up_bed) - today_val
day5_up = (df[df.date == days5]["cases_ub"].values[0] * coeff_up_bed) - day5_val
day10_up= (df[df.date == days10]["cases_ub"].values[0] * coeff_up_bed) - day10_val
day30_up = (df[df.date == days30]["cases_ub"].values[0] * coeff_up_bed) - day30_val

today_low =  today_val - (df[df.date == today]["cases_lb"].values[0] * coeff_low_bed)
day5_low = day5_val - (df[df.date == days5]["cases_lb"].values[0] * coeff_low_bed)
day10_low = day10_val - (df[df.date == days10]["cases_lb"].values[0] * coeff_low_bed)
day30_low =  day30_val - (df[df.date == days30]["cases_lb"].values[0] * coeff_low_bed)

trace0 = go.Bar(
    x=['Today', 'in 5 Days', 'in 10 Days',
       'in 30 Days', 'Capacity'],
    y= [today_val, day5_val, day10_val, day30_val, capacity],
    error_y=dict(
            type='data',
            symmetric=False,
            array=[today_up, day5_up, day10_up, day30_up],
            arrayminus=[today_low, day5_low, day10_low, day30_low]
        ),
    marker=dict(
        color=['#34A853', '#34A853',
               '#34A853', '#34A853',
               '#4285F4']),
)

data_beds = [trace0]
layout_beds = go.Layout(
    margin=go.layout.Margin(
        l=50, #left margin
        r=10, #right margin
        b=50, #bottom margin
        t=10, #top margin
        ))

# ***ICU Bed Bar Chart***
today_val = df[df.date == today]["cases_mean"].values[0] * coeff_icu
day5_val = df[df.date == days5]["cases_mean"].values[0] * coeff_icu
day10_val= df[df.date == days10]["cases_mean"].values[0] * coeff_icu
day30_val = df[df.date == days30]["cases_mean"].values[0] * coeff_icu
capacity = df[df.date == today]["total_ICU_beds"].values[0]

#For Error Bars
today_up = (df[df.date == today]["cases_ub"].values[0] * coeff_up_icu) - today_val
day5_up = (df[df.date == days5]["cases_ub"].values[0] * coeff_up_icu) - day5_val
day10_up= (df[df.date == days10]["cases_ub"].values[0] * coeff_up_icu) - day10_val
day30_up = (df[df.date == days30]["cases_ub"].values[0] * coeff_up_icu) - day30_val

today_low =  today_val - (df[df.date == today]["cases_lb"].values[0] * coeff_low_icu)
day5_low = day5_val - (df[df.date == days5]["cases_lb"].values[0] * coeff_low_icu)
day10_low = day10_val - (df[df.date == days10]["cases_lb"].values[0] * coeff_low_icu)
day30_low =  day30_val - (df[df.date == days30]["cases_lb"].values[0] * coeff_low_icu)

trace0 = go.Bar(
    x=['Today', 'in 5 Days', 'in 10 Days',
       'in 30 Days', 'Capacity'],
    y= [today_val, day5_val, day10_val, day30_val, capacity],
    error_y=dict(
            type='data',
            symmetric=False,
            array=[today_up, day5_up, day10_up, day30_up],
            arrayminus=[today_low, day5_low, day10_low, day30_low]
        ),
    marker=dict(
        color=['#34A853', '#34A853',
               '#34A853', '#34A853',
               '#4285F4']),
)

data_icu = [trace0]
layout_icu = go.Layout(
    margin=go.layout.Margin(
        l=50, #left margin
        r=10, #right margin
        b=50, #bottom margin
        t=10, #top margin
    )
)

# ***Vents Bar Chart***
today_val = df[df.date == today]["cases_mean"].values[0] * coeff_vents
day5_val = df[df.date == days5]["cases_mean"].values[0] * coeff_vents
day10_val= df[df.date == days10]["cases_mean"].values[0] * coeff_vents
day30_val = df[df.date == days30]["cases_mean"].values[0] * coeff_vents
capacity = df[df.date == today]["total_vents"].values[0]

#For Error Bars
today_up = (df[df.date == today]["cases_ub"].values[0] * coeff_up_vents) - today_val
day5_up = (df[df.date == days5]["cases_ub"].values[0] * coeff_up_vents) - day5_val
day10_up= (df[df.date == days10]["cases_ub"].values[0] * coeff_up_vents) - day10_val
day30_up = (df[df.date == days30]["cases_ub"].values[0] * coeff_up_vents) - day30_val


today_low =  today_val - (df[df.date == today]["cases_lb"].values[0] * coeff_low_vents)
day5_low = day5_val - (df[df.date == days5]["cases_lb"].values[0] * coeff_low_vents)
day10_low = day10_val - (df[df.date == days10]["cases_lb"].values[0] * coeff_low_vents)
day30_low =  day30_val - (df[df.date == days30]["cases_lb"].values[0] * coeff_low_vents)



trace0 = go.Bar(
    x=['Today', 'in 5 Days', 'in 10 Days',
       'in 30 Days', 'Capacity'],
    y= [today_val, day5_val, day10_val, day30_val, capacity],
    error_y=dict(
            type='data',
            symmetric=False,
            array=[today_up, day5_up, day10_up, day30_up],
            arrayminus=[today_low, day5_low, day10_low, day30_low]
        ),
    marker=dict(
        color=['#34A853', '#34A853',
               '#34A853', '#34A853',
               '#4285F4']),
)

data_vents = [trace0]
layout_vents = go.Layout(
    margin=go.layout.Margin(
        l=50, #left margin
        r=10, #right margin
        b=50, #bottom margin
        t=10, #top margin
    )
)

# ***Personnel Bar Chart***
today_val = df[df.date == today]["cases_mean"].values[0] * coeff_personnel
day5_val = df[df.date == days5]["cases_mean"].values[0] * coeff_personnel
day10_val= df[df.date == days10]["cases_mean"].values[0] * coeff_personnel
day30_val = df[df.date == days30]["cases_mean"].values[0] * coeff_personnel
capacity = df[df.date == today]["phys_supply"].values[0]

#For Error Bars
today_up = (df[df.date == today]["cases_ub"].values[0] * coeff_up_personnel) - today_val
day5_up = (df[df.date == days5]["cases_ub"].values[0] * coeff_up_personnel) - day5_val
day10_up= (df[df.date == days10]["cases_ub"].values[0] * coeff_up_personnel) - day10_val
day30_up = (df[df.date == days30]["cases_ub"].values[0] * coeff_up_personnel) - day30_val


today_low =  today_val - (df[df.date == today]["cases_lb"].values[0] * coeff_low_personnel)
day5_low = day5_val - (df[df.date == days5]["cases_lb"].values[0] * coeff_low_personnel)
day10_low = day10_val - (df[df.date == days10]["cases_lb"].values[0] * coeff_low_personnel)
day30_low =  day30_val - (df[df.date == days30]["cases_lb"].values[0] * coeff_low_personnel)

trace0 = go.Bar(
    x=['Today', 'in 5 Days', 'in 10 Days',
       'in 30 Days', '# Personnel'],
    y= [today_val, day5_val, day10_val, day30_val, capacity],
    error_y=dict(
            type='data',
            symmetric=False,
            array=[today_up, day5_up, day10_up, day30_up],
            arrayminus=[today_low, day5_low, day10_low, day30_low]
        ),
    marker=dict(
        color=['#34A853', '#34A853',
               '#34A853', '#34A853',
               '#4285F4']),
)

data_personnel = [trace0]
layout_personnel = go.Layout(
    margin=go.layout.Margin(
        l=50, #left margin
        r=10, #right margin
        b=50, #bottom margin
        t=10, #top margin
    )
)

data_disease = [lower_cases, upper_cases, mean_cases, lower_deaths, upper_deaths, mean_deaths]

figure1 = dict(data=data_disease, layout =layout_disease)
figure2 = dict(data=data_beds, layout =layout_beds)
figure3 = dict(data=data_icu, layout =layout_icu)
figure4 = dict(data=data_vents, layout =layout_vents)
figure5 = dict(data=data_personnel, layout =layout_personnel)

# Dash app layout

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
      dbc.DropdownMenu(
            children=[
                    dbc.DropdownMenuItem("Massachusetts", href="/massachusetts"),
                    dbc.DropdownMenuItem("Texas", href="/texas")
            ],
            label="Please select your state",
            color="primary", 
            className="m-1",
            bs_size="md",
            style={'text-align':'center'}
      ),

    html.H1(h1_viz_title, className="app-header", id='pageTitle', 
    style={'text-align':'center','marginTop': '20px', 'marginBottom': '20px'}),
    
    html.Div([
      html.H4('Number of Cases: Statewide',style={'marginLeft': '20px'}),
      html.Div(id='line-chart1'),
      dcc.Graph(figure=figure1)
      ], style={'marginBottom': '50px'}
      ),
     html.Div([
      html.H4('Expected Need: Hospital Beds',style={'marginLeft': '20px'}),
      html.Div(id='bar-chart1'),
      dcc.Graph(figure=figure2)
      ], style={'width': '23%', 'display': 'inline-block'}
      ),
     html.Div([
      html.H4('Expected Need: ICU Beds',style={'marginLeft': '20px'}),
      html.Div(id='bar-chart2'),
      dcc.Graph(figure=figure3)
      ], style={'width': '23%', 'display': 'inline-block'}),
     html.Div([
      html.H4('Expected Need: Ventilators',style={'marginLeft': '20px'}),
      html.Div(id='bar-chart3'),
      dcc.Graph(figure=figure4)
      ], style={'width': '23%', 'display': 'inline-block'}
      ),
     html.Div([
      html.H4('Expected Need: Personnel',style={'marginLeft': '10px'}),
      html.Div(id='bar-chart4'),
      dcc.Graph(figure=figure5)
      ], style={'width': '23%', 'display': 'inline-block'}
      ),
    ])
])
'''
@app.callback(Output(component_id='line', component_property='figure'),
    [ Input(component_id='dropdown-state', component_property='value')])
'''

if __name__ == '__main__':
    app.run_server( host='0.0.0.0', port=os.environ.get('PORT',8050) )