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

# https://www.dontusethiscode.com/blog/2024-03-06_indexes_and_sets.html

from pandas import Series, date_range
from numpy.random import default_rng

rng = default_rng(0)
all_dates = date_range('2000-01-01', '2000-12-31', freq='D', name='date')

df = (
    DataFrame(
        index=all_dates,
        data={
            'temperature': rng.normal(60, scale=5, size=len(all_dates)),
            'precip': rng.uniform(0, 3, size=len(all_dates)), 
        }
    )
    .sample(frac=.99, random_state=rng) # sample 99% of our dataset
    .sort_index()
)

st.write('df',df)
st.write('days missing',
    df.reset_index()
    .merge(
        all_dates.to_frame(index=False), on='date', indicator=True, how='outer'
    )
    .query('_merge == "right_only"')
    ['date']
)

st.write('days missing full merge',
    df.reset_index()
    .merge(
        all_dates.to_frame(index=False), on='date', indicator=True, how='outer'
    ))

st.write('anser',all_dates.difference(df.index))



