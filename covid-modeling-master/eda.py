from collections import defaultdict
import csv
import os

import numpy as np
import pandas as pd

from mass_pop_data import ma_county_pops
from tx_pop_data import tx_county_pops
from us_states import US_STATES

HIGH_PRIORITY_STATES = [
    "Massachusetts",
    "Texas",
    "California",
    "Washington",
    "New York",
    "Hawaii",
    "Alaska",
    "Guam"
]



def none_fill(xs):
    return [x if x > 0 else None for x in xs]

if not "confirmed_df" in globals():
    print("DLing confirmed")
    confirmed_df = pd.read_csv(
        "https://www.soothsawyer.com/wp-content/uploads/2020/03/time_series_19-covid-Confirmed.csv"
    )
if not "deaths_df" in globals():
    print("DLing deaths")
    deaths_df = pd.read_csv(
        "https://www.soothsawyer.com/wp-content/uploads/2020/03/time_series_19-covid-Deaths.csv"
    )

confirmed_date = confirmed_df.columns[-1]
deaths_date = deaths_df.columns[-1]
assert(confirmed_date == deaths_date)

print("Confirmed df date:", confirmed_date)
print("Deaths df date:", deaths_df.columns[-1])
def interpret_df(df):
    # template = "../COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-{}.csv"
    # fname = template.format(data_kind)
    # with open(fname) as f:
    #     lines = [line for line in csv.reader(f)]
    results = {}
    for i, line in df.iterrows():
        if i == 0:
            continue
        try:
            province = line[0]
            country = line[1]
            lat = line[2]
            lon = line[3]
            time_series = [int(x) if pd.notnull(x) else x for x in line[4:]]
            results[province, country] = time_series
        except Exception as e:
            print("FAILED on:", e, line)

    return results

# confirmed = interpret_df(confirmed_df)
# deaths = interpret_df(deaths_df)

def aggregate_provinces(d):
    N = min(len(time_series) for time_series in (d.values()))
    d_out = defaultdict(lambda: np.zeros(N))
    for province_country, time_series in d.items():
        province, country = province_country
        d_out[country] += np.array(time_series)[:N]
    return dict(d_out)

def show_data(d):
    for k, v in d.items():
        plt.plot(none_fill(v), label=k)
    plt.legend()
    plt.semilogy()
    plt.show()

us_confirmed_df = confirmed_df[
    confirmed_df['Province/State'].isin(US_STATES) &
    (confirmed_df['Country/Region'] == 'US')
]
us_deaths_df = deaths_df[
    deaths_df['Province/State'].isin(US_STATES) &
    (deaths_df['Country/Region'] == 'US')
]

date_cols = us_confirmed_df.columns[4:]  # time-series data containing cols
us_data = {
    'confirmed': np.sum(us_confirmed_df[date_cols]).values,
    'deaths': np.sum(us_deaths_df[date_cols]).values,
    #'recovered': aggregate_provinces(recovered)['US'],
    'pop': 3.27 * 10**8
}

italy_data = {
    'confirmed':confirmed_df[confirmed_df['Country/Region'] == 'Italy'][date_cols].values[0],
    'deaths':deaths_df[deaths_df['Country/Region'] == 'Italy'][date_cols].values[0],
    'pop': 60.48 * 10**6
}
is_MA = confirmed_df['Province/State'] == 'Massachusetts'
ma_data = {
    'confirmed': confirmed_df[is_MA][date_cols].values[0],
    'deaths': deaths_df[is_MA][date_cols].values[0],
    #'recovered': np.array(recovered[('Massachusetts', 'US')]),
    'pop': sum(ma_county_pops.values())

}
is_TX = confirmed_df['Province/State'] == 'Texas'
tx_data = {
    'confirmed': confirmed_df[is_TX][date_cols].values[0],
    'deaths': deaths_df[is_TX][date_cols].values[0],
    #'recovered': np.array(recovered[('Massachusetts', 'US')]),
    'pop': sum(tx_county_pops.values())

}
def plot_us():
    us_confirmed = aggregate_provinces(confirmed)['US']
    us_deaths = aggregate_provinces(deaths)['US']
    us_recovered = aggregate_provinces(recovered)['US']

    plt.plot(none_fill(us_confirmed), label='confirmed')
    plt.plot(none_fill(us_deaths), label='deaths')
    plt.plot(none_fill(us_recovered), label='recovered')
    plt.legend()
    plt.semilogy()
    plt.show()
