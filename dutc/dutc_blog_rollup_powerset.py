# sourcery skip: merge-list-append
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



# https://www.dontusethiscode.com/blog/2024-06-12_groupby-sets.html

rng = default_rng(0)

center_ids = [f"Center_{i}" for i in range(2)]
locations = ["East", "West"]
service_types = ["Web Hosting", "Data Processing", "Cloud Storage"]

pd_df = DataFrame({
    "center_id":    rng.choice(center_ids, size=(size := 100)),
    "location":     rng.choice(locations, size=size),
    "service_type": rng.choice(service_types, size=size),
    "timestamp":    Timestamp.now() - to_timedelta(rng.integers(0, 3_650, size=size), unit='D'),
    "cpu_usage":    rng.uniform(0, 100, size=size),
    "mem_usage":    rng.uniform(0, 64, size=size),
})

st.write('pd df', pd_df)

results = []
aggfuncs = {'cpu_usage': 'mean', 'mem_usage': 'mean'}
groupings = [
    ['center_id', 'location', 'service_type'],
    ['service_type'],
    [],
]

for gs in groupings:
    if not gs:
        res = pd_df.agg(aggfuncs).to_frame().T
    else:
        res = pd_df.groupby(gs, as_index=False).agg(aggfuncs)
    results.append(res)
    
st.write('result',pd_concat(results, ignore_index=True))
# for gs in groupings:
#     st.write('this is gs within for loop',gs)
# st.write('this is res within for loop',pd_df.groupby('center_id', as_index=False).agg(aggfuncs))

test_results=[]
test_results.append(pd_df.groupby('service_type', as_index=False).agg(aggfuncs))
st.write('1 test results service type', test_results)
test_results.append(pd_df.groupby(['center_id', 'location', 'service_type'], as_index=False).agg(aggfuncs))
st.write('2 test results x 3 types', test_results)
# test_results.append(pd_df.groupby([]], as_index=False).agg(aggfuncs))
# st.write('3 test results', test_results)
# test_results.append(pd_df.groupby('service_type', as_index=False).agg(aggfuncs))
# st.write('4 test results', test_results)


def rollup(*items):
    "reversed version of itertools powerset recipe"
    return (
        [*islice(items, i, None)] for i in range(len(items)+1)
    )


st.write('rollup',rollup('center_id', 'location', 'service_type'))

# for gs in rollup('center_id', 'location', 'service_type'):
#     st.write(gs)


from pandas import concat, Series as pd_Series

results = []
aggfuncs = {'cpu_usage': 'mean', 'mem_usage': 'mean'}
for gs in rollup('center_id', 'location', 'service_type'):
    if not gs:
        res = pd_df.agg(aggfuncs).to_frame().T
    else:
        res = pd_df.groupby(list(gs), as_index=False).agg(aggfuncs)
    results.append(
        res.assign(
            groupings=lambda d: pd_Series([gs] * len(d), index=d.index)
        )
    )
    
st.write('result of the rollup',concat(results, ignore_index=True))

from itertools import chain, combinations

def cube(*items):
    """reversed version of itertools powerset recipe"""
    for size in range(len(items)+1, -1, -1):
        for combo in combinations(items, size):
            yield [*combo]

for gs in cube('center_id', 'location', 'service_type'):
    st.write(gs)

from pandas import concat, Series as pd_Series

results = []
aggfuncs = {'cpu_usage': 'mean', 'mem_usage': 'mean'}
for gs in cube('center_id', 'location', 'service_type'):
    if not gs:
        res = pd_df.agg(aggfuncs).to_frame().T
    else:
        res = pd_df.groupby(list(gs), as_index=False).agg(aggfuncs)
    results.append(
        res.assign(
            groupings=lambda d: pd_Series([gs] * len(d), index=d.index)
        )
    )
    
st.write(concat(results, ignore_index=True).fillna('')) # fillna to view output easily


