import pandas as pd
import numpy as np
from numpy import unique,where,array,tile
import streamlit as st
from pandas import DataFrame,to_datetime,MultiIndex,period_range,IndexSlice, Series,merge,concat, get_dummies, NA, Timestamp, to_timedelta, concat as pd_concat, CategoricalIndex,date_range
from collections import Counter
from itertools import islice, groupby, pairwise, cycle, tee, zip_longest, chain, repeat, takewhile, product
from numpy.random import default_rng
from datetime import timedelta
from random import Random
from enum import Enum
from string import ascii_lowercase, digits

st.set_page_config(layout="wide")

# https://www.youtube.com/watch?v=Bj8yyS1o_hI
# Deltas - State DUTC JP video


rng=default_rng(0)
rnd=Random(0)

State = Enum ('State', 'Provisioned Deployed InUse Maintenance Decommissioned')

def simulate():
    td = lambda days: timedelta(seconds=rnd.randrange(24*60*60*days))

    yield State.Provisioned, td(days=90)

    if rnd.choices([True, False], weights = [.90,.10])[0]:
        yield State.Deployed, td(days=30)

        if rnd.choices([True, False], weights = [.95,.05])[0]:
                for _ in range(max(0, int(rnd.gauss(mu=2, sigma=1)))):
                     yield State.InUse, td(days=14)
                     yield State.Maintenance, td(days=14)
                if rnd.choices([True,False], weights = [.95,.05])[0]:
                     yield State.InUse, td(days=14)
             
        if rnd.choices([True, False], weights = [.50,.50])[0]:
             yield State.Decommissioned, td(days=30)

if __name__ == '__main__':
     devices = rng.choice([*ascii_lowercase, *digits], size= (10,9))
     devices[:,4] = '-'
     devices = devices.view('<U9').ravel()

     all_lifecycles = []
     for d in devices:
          # Create DataFrame from the simulation
          lifecycle_df = DataFrame(simulate(), columns=['state', 'time'])
          lifecycle_df['device'] = d
          
          # Calculate cumulative times before setting index
          lifecycle_df['time'] = lifecycle_df['time'].cumsum()
          
          # Convert time to datetime
          lifecycle_df['time'] = to_datetime('2020-01-01') + lifecycle_df['time']

          # Store the enum name instead of the enum object
          lifecycle_df['state'] = lifecycle_df['state'].apply(lambda x: x.name) # to do with the Enum and saving to csv claude helped with this

          # Set index after calculations
          lifecycle_df = lifecycle_df.set_index(['device', 'state'])
          
          # Append to our collection
          all_lifecycles.append(lifecycle_df)

          # Concatenate all device lifecycles
          lifecycles = concat(all_lifecycles)
          # Store the enum name instead of the enum object
          # lifecycles['state'] = lifecycles['state'].apply(lambda x: x.name)

# st.write('lifecycles', lifecycles) 
lifecycles.to_csv('C:/Users/Darragh/Documents/Python/dutc/lifecycles.csv')
     # read csv file
df_csv = pd.read_csv('C:/Users/Darragh/Documents/Python/dutc/lifecycles.csv')
# Convert state strings back to Enum values
df_csv['state'] = df_csv['state'].apply(lambda x: State[x])
# Reset index if it was stored as index in the CSV
if 'device' in df_csv.columns and 'state' in df_csv.columns:
    pass  # Index was already reset when saving
else:
    df_csv = df_csv.reset_index()
st.write('df csv',df_csv)
st.write('this is james code',
     df_csv.reset_index('state', drop=False).set_index('time',append=True)
     .groupby('device').agg(lambda g: sum(1 for x in g if x is State.Maintenance))
)

maintenance_counts = (
    df_csv.reset_index()  # Reset both 'device' and 'state' indices
    .set_index(['device', 'time'])   # Set new index
    .groupby('device')['state']      # Group by device and select the state column
    .apply(lambda states: sum(1 for state in states if state == State.Maintenance))
)
st.write('claude 1',maintenance_counts)

maintenance_counts_alt = (
    df_csv.reset_index()
    .query("state == @state.Maintenance")  # Filter for maintenance states
    .groupby('device').size()              # Count occurrences by device
)

st.write('claude 2',maintenance_counts_alt)

     # lifecycles = (
     #    DataFrame(simulate(), columns = 'state time'.split())
     #            .assign(device=d)
     #            .set_index(['device','state'])
     #            .cumsum()
     #            .pipe(lambda df: df + to_datetime('2020-01-01'))
     #            .squeeze(axis='columns')
     #    for d in devices
     # )
     # lifecycles = concat( lifecycles, axis='index')

