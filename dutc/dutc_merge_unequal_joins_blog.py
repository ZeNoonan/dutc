from pandas import read_csv, DataFrame
from io import StringIO
from textwrap import dedent
import pandas as pd
import numpy as np
from numpy import unique,where,array,tile
import streamlit as st
from pandas import DataFrame,MultiIndex,period_range,IndexSlice, Series,merge,concat, get_dummies, NA, Timestamp, to_timedelta, concat as pd_concat, CategoricalIndex,date_range
from string import ascii_lowercase
from collections import Counter
from itertools import islice, groupby, pairwise, cycle, tee, zip_longest, chain, repeat, takewhile, product
from numpy.random import default_rng
from io import StringIO
from textwrap import dedent
from csv import reader
from collections import deque
st.set_page_config(layout="wide")

# https://www.dontusethiscode.com/blog/2024-07-10_ineq_joins.html

from pandas import DataFrame, Timedelta

state_polling_df = DataFrame(
    columns=[      'timestamp', 'device_id',  'state'],
    data=[
        ['2000-01-01 04:00:00',       'abc', 'state1'],
        ['2000-01-01 04:30:00',       'abc', 'state1'],
        ['2000-01-01 05:00:00',       'abc', 'state1'],
        ['2000-01-01 05:30:00',       'abc', 'state3'],
        ['2000-01-01 06:00:00',       'abc', 'state3'],
        ['2000-01-01 06:30:00',       'abc', 'state2'],
        ['2000-01-01 07:00:00',       'abc', 'state2'],
        ['2000-01-01 07:30:00',       'abc', 'state1'],

        ['2000-01-01 04:00:00',       'def', 'state2'],
        ['2000-01-01 04:30:00',       'def', 'state1'],
        ['2000-01-01 05:00:00',       'def', 'state3'],
        ['2000-01-01 05:30:00',       'def', 'state3'],
        ['2000-01-01 06:00:00',       'def', 'state1'],
        ['2000-01-01 06:30:00',       'def', 'state1'],
    ]
).astype({'timestamp': 'datetime64[ns]'})

alert_events = DataFrame(
    columns=[      'timestamp', 'device_id'],
    data=[
        ['2000-01-01 03:15:00',       'abc'],
        ['2000-01-01 04:05:00',       'abc'],
        ['2000-01-01 04:17:00',       'abc'],
        ['2000-01-01 04:44:00',       'abc'],
        ['2000-01-01 05:10:00',       'abc'],
        ['2000-01-01 05:23:00',       'abc'],
        ['2000-01-01 05:43:00',       'abc'],
        ['2000-01-01 05:53:00',       'abc'],
        ['2000-01-01 06:02:00',       'abc'],
        ['2000-01-01 06:08:00',       'abc'],
        ['2000-01-01 06:10:00',       'abc'],
        ['2000-01-01 06:23:00',       'abc'],
        ['2000-01-01 06:51:00',       'abc'],

        ['2000-01-01 03:05:00',       'def'],
        ['2000-01-01 04:15:00',       'def'],
        ['2000-01-01 04:27:00',       'def'],
        ['2000-01-01 04:34:00',       'def'],
        ['2000-01-01 05:20:00',       'def'],
        ['2000-01-01 05:33:00',       'def'],
        ['2000-01-01 06:22:00',       'def'],
        ['2000-01-01 06:29:00',       'def'],
        ['2000-01-01 06:43:00',       'def'],
        ['2000-01-01 07:01:00',       'def'],
    ]
).astype({'timestamp': 'datetime64[ns]'})

st.write(state_polling_df.head(), alert_events.head())
# state_polling_df.head(3)

state_df = (
    state_polling_df
    .assign(
        state_shift=lambda d:
            d.groupby('device_id')['state'].shift() != d['state'],
        state_group=lambda d: d.groupby('device_id')['state_shift'].cumsum(),
    )
    .groupby(['device_id', 'state', 'state_group'], as_index=False)
    .agg(
        start=('timestamp', 'min'),
        stop= ('timestamp', 'max'),
    )
)

st.write(state_df.sort_values(['device_id', 'start']))

