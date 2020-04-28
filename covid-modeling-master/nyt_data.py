import numpy as np
import pandas as pd

county_pops = pd.read_csv("us_county_populations.csv", dtype={'population':int}, thousands=',')
nyt_df = pd.read_csv(
    "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv",
    dtype={'fips': pd.Int64Dtype()}  # fips is a nullable int
)
EPOCH = pd.to_datetime(nyt_df.date.min())

county_state_pairs = sorted(set([tuple(x) for x in nyt_df[['county', 'state']].values]))
county_data = {}
# TODO: Deal with one missing value for KC, MO
for county, state in county_state_pairs:
    if county == 'Unknown':
        # TODO: ~2% of all cases unassigned to county.  skip for now.
        continue

    county_df = nyt_df[(nyt_df['county'] == county) & (nyt_df['state'] == state)]
    if not (all(county_df.date.values == sorted(county_df.date))):
        print("Failed for improper sorting:", county, state )
        continue
    start = pd.to_datetime(county_df.date.iloc[0])
    end = pd.to_datetime(county_df.date.iloc[-1])
    delta = (end - start).days
    if not (len(county_df) == delta + 1):
        print("Failed for missing data:", county, state)
        continue
    pad = np.zeros((start - EPOCH).days)
    try:
        pop = int(county_pops[
            (county_pops.county == county) & (county_pops.state == state)
        ].population)
    except:
        print("Failed pop on:", county, state)
    this_county_data = {
        'cases': np.hstack([pad, county_df.cases.values]),
        'deaths': np.hstack([pad, county_df.deaths.values]),
        'pop': pop
    }
    county_data[(county, state)] = this_county_data
