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

# Instead of relying on explicit .groupby operations you can create your groups in Python. 
# Considering these operations are performed along each column this should be quite performant as well 
# and does not require a possibly expensive transpose or a deprecate API.
# The idea here is to use itertools to create the groupings, store the intermediate results in a dictionary and recreate a new DataFrame from those parts.

# https://stackoverflow.com/questions/78692255/merge-dataframe-based-on-substring-column-labeld-while-keep-the-original-columns/78693761#78693761

df=pd.DataFrame({
    "[RATE] BOJ presser/2024-03-19T07:30:00Z/2024-03-19T10:30:00Z": [1],
    "[RATE] BOJ/2024-01-23T04:00:00Z/2024-01-23T07:00:00Z": [2],
    "[RATE] BOJ/2024-03-19T04:00:00Z/2024-03-19T07:00:00Z": [3],
    "[RATE] BOJ/2024-04-26T03:00:00Z/2024-04-26T06:00:00Z": [4],
    "[RATE] BOJ/2024-04-26T03:00:00Z/2024-04-26T08:00:00Z": [5],
    "[RATE] BOJ/2024-06-14T03:00:00Z/2024-06-14T06:00:00Z": [6],
    "[RATE] BOJ/2024-06-14T03:00:00Z/2024-06-14T08:00:00Z": [7],
    "[RATE] BOJ/2024-07-31T03:00:00Z/2024-07-31T06:00:00Z": [8],
    "[RATE] BOJ/2024-07-31T03:00:00Z/2024-07-31T08:00:00Z": [9],
    "[RATE] BOJ/2024-09-20T03:00:00Z/2024-09-20T06:00:00Z": [10],
    "[RATE] BOJ/2024-09-20T03:00:00Z/2024-09-20T08:00:00Z": [11],
    "[RATE] BOJ/2024-10-31T04:00:00Z/2024-10-31T07:00:00Z": [12],
    "[RATE] BOJ/2024-10-31T04:00:00Z/2024-10-31T09:00:00Z": [13],
    "[RATE] BOJ/2024-12-19T04:00:00Z/2024-12-19T07:00:00Z": [14],
    "[RATE] BOJ/2024-12-19T04:00:00Z/2024-12-19T09:00:00Z": [15],
})

st.write('dataframe', df.T)

result_want=pd.DataFrame({
        "[RATE] BOJ presser/2024-03-19T07:30:00Z/2024-03-19T10:30:00Z": [1],
        "[RATE] BOJ/2024-01-23T04:00:00Z/2024-01-23T07:00:00Z": [2],
        "[RATE] BOJ/2024-03-19T04:00:00Z/2024-03-19T07:00:00Z": [3],
        "[RATE] BOJ/2024-04-26T03:00:00Z/2024-04-26T06:00:00Z": [9],
        "[RATE] BOJ/2024-06-14T03:00:00Z/2024-06-14T06:00:00Z": [13],
        "[RATE] BOJ/2024-07-31T03:00:00Z/2024-07-31T06:00:00Z": [17],
        "[RATE] BOJ/2024-09-20T03:00:00Z/2024-09-20T06:00:00Z": [21]
    })

st.write('result',result_want.T)

groupings = (
    df.columns.str.extract(r'([^/]+)/(\d{4}-\d{2}-\d{2})')
)
unique = groupings.assign(orig=df.columns).drop_duplicates([0, 1])

result = (
    df.T
    .groupby([col.values for _, col in groupings.items()]).sum()
    .set_axis(unique['orig']).T
)

st.write('first solution',result.T)

def extract_unique(column):
    splitted = column.split('/')
    return splitted[0], splitted[1][:10]

# sourcery skip: merge-list-append
result = {}
for _, col_group in groupby(sorted(df.columns), key=extract_unique):
    first, *remaining = col_group
    result[first] = df[[first, *remaining]].sum(axis=1)

result = pd.DataFrame(result)
st.write('2nd solution',result.T)