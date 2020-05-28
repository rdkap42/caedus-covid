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
from flask import render_template
import plotly.graph_objs as go
from plotly.offline import iplot
import gcsfs
import locale

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
print(data_file)
df_all = pd.read_csv(f'{data_file}')

#locale.setlocale(locale.LC_ALL, 'en_US.utf8')
#TODO https://docs.python.org/3/library/locale.html
df_all['Date']= pd.to_datetime(df_all['Date'])
#df_all['State']=df_all['State'].astype('string')
#df =  df_all[((df_all['Date']) < (today + np.timedelta64(150, 'D'))) & (df_all['State'] == 'MA')]
df = df_all[(df_all['Date']) < (today + np.timedelta64(150, 'D'))]
df = df[['Date', 'State','Cases_Mean', 'Cases_LB', 'Cases_UB', 'Deaths_Mean', 'Deaths_LB', 
'Deaths_UB', 'total_beds', 'total_ICU_beds', 'total_vents', 'phys_supply']]
df.columns = ['date','state','cases_mean', 'cases_lb', 'cases_ub', 'deaths_mean', 'deaths_lb', 
'deaths_ub', 'total_beds', 'total_ICU_beds', 'total_vents', 'phys_supply']

# function to generate the no data alert message
def noData():
    return dbc.Alert("We do not have data yet for this state. Please select another state!", color="warning")

# function to generates graphs
def generate_viz(dff):
    # ***Disease Curve Line Chart***
    lower_cases = go.Scatter(
        x=dff.date,
        y=dff.cases_lb,
        fill= None,
        mode='lines',
        showlegend = False,
        name = "95% CI cases",
        legendgroup="group1",
        line=dict(
        color='#FDD663',
        width = 0,
        )
    )
    upper_cases = go.Scatter(
        x=dff.date,
        y=dff.cases_ub,
        fill='tonexty',
        mode='lines',
        name = "95% CI cases",
        legendgroup="group1",
        line=dict(
        color='#FDD663',
        width = 0
        )
    )
    mean_cases = go.Scatter(
        x=dff.date,
        y=dff.cases_mean,
        mode='lines+markers',
        name = "mean daily cases",
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
        name = "95% CI deaths",
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
        name = "mean total deaths",
        line=dict(
            color='#EA4335',
        )
    )
    layout_disease_cases = go.Layout(
    xaxis = dict(
        showline=True,
        showgrid=True
    ),
    yaxis = dict(
        title = 'Estimated Daily Infections',
        showline=True,
        showgrid=True,
        fixedrange=Auto
    ),
    margin=go.layout.Margin(
            l=100, #left margin
            r=50, #right margin
            b=50, #bottom margin
            t=10, #top margin
        ),
    showlegend=True,
    transition={
        'duration': 1000,
        'easing': 'exp',
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
        text = locale.format_string("%d", int(dff[dff.date == today]["cases_mean"].values[0]), grouping=True),
        showarrow = True,
        arrowhead=7,
        ax = -30,
        ay = -30,
        #clicktoshow = "onout",
        font=dict(
            size=12
        ),
        )
    ]
    )
    layout_disease_deaths = go.Layout(
    xaxis = dict(
        showline=True,
        showgrid=True
    ),
    yaxis = dict(
        title = 'Estimated Total Deaths',
        showline=True,
        showgrid=True,
        fixedrange=Auto
    ),
    margin=go.layout.Margin(
            l=100, #left margin
            r=50, #right margin
            b=50, #bottom margin
            t=10, #top margin
        ),
    showlegend=True,
    transition={
        'duration': 1000,
        'easing': 'exp',
    },
    legend=dict(
        orientation="h"
    ),
    )
    '''
    # ***Hospital Bed Bar Chart***
    today_val = dff[dff.date == today]["cases_mean"].values[0] * coeff_bed
    day5_val = dff[dff.date == days5]["cases_mean"].values[0] * coeff_bed
    day10_val= dff[dff.date == days10]["cases_mean"].values[0] * coeff_bed
    day30_val = dff[dff.date == days30]["cases_mean"].values[0] * coeff_bed
    capacity = dff[dff.date == today]["total_beds"].values[0]

    # For Error Bars
    today_up = (dff[dff.date == today]["cases_ub"].values[0] * coeff_up_bed) - today_val
    day5_up = (dff[dff.date == days5]["cases_ub"].values[0] * coeff_up_bed) - day5_val
    day10_up= (dff[dff.date == days10]["cases_ub"].values[0] * coeff_up_bed) - day10_val
    day30_up = (dff[dff.date == days30]["cases_ub"].values[0] * coeff_up_bed) - day30_val

    today_low =  today_val - (dff[dff.date == today]["cases_lb"].values[0] * coeff_low_bed)
    day5_low = day5_val - (dff[dff.date == days5]["cases_lb"].values[0] * coeff_low_bed)
    day10_low = day10_val - (dff[dff.date == days10]["cases_lb"].values[0] * coeff_low_bed)
    day30_low =  day30_val - (dff[dff.date == days30]["cases_lb"].values[0] * coeff_low_bed)

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
    today_val = dff[dff.date == today]["cases_mean"].values[0] * coeff_icu
    day5_val = dff[dff.date == days5]["cases_mean"].values[0] * coeff_icu
    day10_val= dff[dff.date == days10]["cases_mean"].values[0] * coeff_icu
    day30_val = dff[dff.date == days30]["cases_mean"].values[0] * coeff_icu
    capacity = dff[dff.date == today]["total_ICU_beds"].values[0]

    #For Error Bars
    today_up = (dff[dff.date == today]["cases_ub"].values[0] * coeff_up_icu) - today_val
    day5_up = (dff[dff.date == days5]["cases_ub"].values[0] * coeff_up_icu) - day5_val
    day10_up= (dff[dff.date == days10]["cases_ub"].values[0] * coeff_up_icu) - day10_val
    day30_up = (dff[dff.date == days30]["cases_ub"].values[0] * coeff_up_icu) - day30_val

    today_low =  today_val - (dff[dff.date == today]["cases_lb"].values[0] * coeff_low_icu)
    day5_low = day5_val - (dff[dff.date == days5]["cases_lb"].values[0] * coeff_low_icu)
    day10_low = day10_val - (dff[dff.date == days10]["cases_lb"].values[0] * coeff_low_icu)
    day30_low =  day30_val - (dff[dff.date == days30]["cases_lb"].values[0] * coeff_low_icu)

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
    today_val = dff[dff.date == today]["cases_mean"].values[0] * coeff_vents
    day5_val = dff[dff.date == days5]["cases_mean"].values[0] * coeff_vents
    day10_val= dff[dff.date == days10]["cases_mean"].values[0] * coeff_vents
    day30_val = dff[dff.date == days30]["cases_mean"].values[0] * coeff_vents
    capacity = dff[dff.date == today]["total_vents"].values[0]

    #For Error Bars
    today_up = (dff[dff.date == today]["cases_ub"].values[0] * coeff_up_vents) - today_val
    day5_up = (dff[dff.date == days5]["cases_ub"].values[0] * coeff_up_vents) - day5_val
    day10_up= (dff[dff.date == days10]["cases_ub"].values[0] * coeff_up_vents) - day10_val
    day30_up = (dff[dff.date == days30]["cases_ub"].values[0] * coeff_up_vents) - day30_val


    today_low =  today_val - (dff[dff.date == today]["cases_lb"].values[0] * coeff_low_vents)
    day5_low = day5_val - (dff[dff.date == days5]["cases_lb"].values[0] * coeff_low_vents)
    day10_low = day10_val - (dff[dff.date == days10]["cases_lb"].values[0] * coeff_low_vents)
    day30_low =  day30_val - (dff[dff.date == days30]["cases_lb"].values[0] * coeff_low_vents)



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
    today_val = dff[dff.date == today]["cases_mean"].values[0] * coeff_personnel
    day5_val = dff[dff.date == days5]["cases_mean"].values[0] * coeff_personnel
    day10_val= dff[dff.date == days10]["cases_mean"].values[0] * coeff_personnel
    day30_val = dff[dff.date == days30]["cases_mean"].values[0] * coeff_personnel
    capacity = dff[dff.date == today]["phys_supply"].values[0]

    #For Error Bars
    today_up = (dff[dff.date == today]["cases_ub"].values[0] * coeff_up_personnel) - today_val
    day5_up = (dff[dff.date == days5]["cases_ub"].values[0] * coeff_up_personnel) - day5_val
    day10_up= (dff[dff.date == days10]["cases_ub"].values[0] * coeff_up_personnel) - day10_val
    day30_up = (dff[dff.date == days30]["cases_ub"].values[0] * coeff_up_personnel) - day30_val


    today_low =  today_val - (dff[dff.date == today]["cases_lb"].values[0] * coeff_low_personnel)
    day5_low = day5_val - (dff[dff.date == days5]["cases_lb"].values[0] * coeff_low_personnel)
    day10_low = day10_val - (dff[dff.date == days10]["cases_lb"].values[0] * coeff_low_personnel)
    day30_low =  day30_val - (dff[dff.date == days30]["cases_lb"].values[0] * coeff_low_personnel)

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
    '''
    data_disease_cases = [lower_cases, upper_cases, mean_cases]
    figure1 = dict(data=data_disease_cases, layout =layout_disease_cases)
    data_disease_deaths = [lower_deaths, upper_deaths, mean_deaths]
    figure2 = dict(data=data_disease_deaths, layout=layout_disease_deaths)
    # Hide all bar chart grpahs
    #figure3 = dict(data=data_beds, layout =layout_beds)
    #figure4 = dict(data=data_icu, layout =layout_icu)
    #figure5 = dict(data=data_vents, layout =layout_vents)
    #figure6 = dict(data=data_personnel, layout =layout_personnel)
    
    return html.Div([html.Div([
      html.H4('Estimated Daily Cases Statewide'),
      html.Div(id='line-chart1'),
      dcc.Graph(figure=figure1, config={'displayModeBar': False}),
      html.H4('Estimated Total Deaths Statewide'),
      dcc.Graph(figure=figure2, config={'displayModeBar': False})
      ]
      ),
  ])

# App flask config
server = Flask(__name__)
server.debug = True

app = Dash(__name__, 
    server=server,
    external_stylesheets=[dbc.themes.LUX], 
    url_base_pathname='/')
app.config['suppress_callback_exceptions']=True

# Dash app layout
app.layout = html.Div([
    html.Div([
      dcc.Dropdown(
        id='state-dropdown',
        options=[
            {'label': 'Alabama', 'value': '1'},
            {'label': 'Alaska', 'value': '2'},
            {'label': 'Arizona', 'value': '3'},
            {'label': 'Arkansas', 'value': '4'},
            {'label': 'California', 'value': '5'},
            {'label': 'Colorado', 'value': '6'},
            {'label': 'Connecticut', 'value': '7'},
            {'label': 'Delaware', 'value': '8'},
            {'label': 'District of Columbia', 'value': '9'},
            {'label': 'Florida', 'value': '10'},
            {'label': 'Georgia', 'value': '11'},
            {'label': 'Hawaii', 'value': '12'},
            {'label': 'Idaho', 'value': '13'},
            {'label': 'Illinios', 'value': '14'},
            {'label': 'Indiana', 'value': '15'},
            {'label': 'Iowa', 'value': '16'},
            {'label': 'Kansas', 'value': '17'},
            {'label': 'Kentucky', 'value': '18'},
            {'label': 'Louisiana', 'value': '19'},
            {'label': 'Maine', 'value': '20'},
            {'label': 'Maryland', 'value': '21'},
            {'label': 'Massachusetts', 'value': '22'},
            {'label': 'Michigan', 'value': '23'},
            {'label': 'Minnesota', 'value': '24'},
            {'label': 'Mississippi', 'value': '25'},
            {'label': 'Missouri', 'value': '26'},
            {'label': 'Montana', 'value': '27'},
            {'label': 'Nebraska', 'value': '28'},
            {'label': 'Nevada', 'value': '29'},
            {'label': 'New Hampshire', 'value': '30'},
            {'label': 'New Jersey', 'value': '31'},
            {'label': 'New Mexico', 'value': '32'},
            {'label': 'New York', 'value': '33'},
            {'label': 'North Carolina', 'value': '34'},
            {'label': 'North Dakota', 'value': '35'},
            {'label': 'Ohio', 'value': '36'},
            {'label': 'Oklahoma', 'value': '37'},
            {'label': 'Oregon', 'value': '38'},
            {'label': 'Pennsylvania', 'value': '39'},
            {'label': 'Rhode Island', 'value': '40'},
            {'label': 'South Carolina', 'value': '41'},
            {'label': 'South Dakota', 'value': '42'},
            {'label': 'Tennessee', 'value': '43'},
            {'label': 'Texas', 'value': '44'},
            {'label': 'Utah', 'value': '45'},
            {'label': 'Vermont', 'value': '46'},
            {'label': 'Virginia', 'value': '47'},
            {'label': 'Washington', 'value': '48'},
            {'label': 'West Virginia', 'value': '49'},
            {'label': 'Wisconsin', 'value': '50'},
            {'label': 'Wyoming', 'value': '51'}
    ],
    style={'marginBottom': '30px', 'marginTop': '10px', 
    'text-align': 'center',
    'text-transform': 'uppercase',
    'font-size':'100%',
    'height': '30px'},
    placeholder="Select a state",
    value=[""], # default value
    )]),
    
    html.Div(id='viz-div', children=[]),
    ])
@app.callback(
    dash.dependencies.Output('viz-div', 'children'),
    [dash.dependencies.Input('state-dropdown', 'value')])
def display_dcurve(value): 
    dff = df[df.state.isin([value])]
    print(value)
    if dff.empty == True:
        return noData()
    else:
        return generate_viz(dff)

if __name__ == '__main__':
    app.run_server( host='0.0.0.0', port=os.environ.get('PORT',8050), dev_tools_hot_reload=False)