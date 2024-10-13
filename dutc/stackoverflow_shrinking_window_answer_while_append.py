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

# https://stackoverflow.com/questions/77781290/filter-values-within-a-percentage-range-based-on-the-last-non-filtered-value/77781874#77781874


df = pd.DataFrame(
    {'trade': [100, NA, NA, 101, NA, 102, NA, 98, 107, NA, 101, NA, 98, NA, NA, 94]}
).astype({'trade': 'Int32'})

st.write('DF',df)

tmp = df.dropna()
st.write('after dropna', tmp)
valid = [0]
st.write('tmp.index[-1]',)
while valid[-1] < tmp.index[-1]:
    st.write('valid[-1]',valid[-1])
    st.write('tmp.index[-1]',tmp.index[-1])
    st.write("chunk:  =  tmp.loc[valid[-1]:, 'trade']",tmp.loc[valid[-1]:, 'trade'])
    chunk = tmp.loc[valid[-1]:, 'trade'] # get window of all unprocessed data
    st.write('chunk.iat[0]',chunk.iat[0])
    target = chunk.iat[0]
    valid.append(                        # find the first boundary
        chunk.between(target * .95, target * 1.05).idxmin() 
    )



st.write(
    f'{valid = }',  # [0, 8, 10, 15] (while loop took len(valid) iterations)
    df.assign(      # mask over values not in `valid`
        cleaned=lambda d: d['trade'].where(d.index.isin(valid)),
    ),
    sep='\n\n',
)

df=df.assign(cleaned=lambda d: d['trade'].where(d.index.isin(valid)))
st.write(df)
# valid = [0, 8, 10, 15]
#
#     trade  cleaned
# 0     100      100
# 1    <NA>     <NA>
# 2    <NA>     <NA>
# 3     101     <NA>
# 4    <NA>     <NA>
# 5     102     <NA>
# 6    <NA>     <NA>
# 7      98     <NA>
# 8     107      107
# 9    <NA>     <NA>
# 10    101      101
# 11   <NA>     <NA>
# 12     98     <NA>
# 13   <NA>     <NA>
# 14   <NA>     <NA>
# 15     94       94


# questioners answer interesting
stop = 0.05
# Prepare
df = pd.DataFrame(
    {'Trade': [100, NA, NA, 101, NA, 102, NA, 98, 107, NA, 101, NA, 98, NA, NA, 94]}
).astype({'Trade': 'Int32'})
trade = df['Trade']
trade.fillna(0, inplace=True)
trade = trade[trade != 0]

# Filter
trade.loc[(trade > trade.shift(1) * (1-stop)) & 
 (trade < trade.shift(1) * (1+stop))] = 0

# Recompose Dataframe
df = df.merge(trade.rename('T1'), left_index=True, 
right_index=True, how='left')
df['Trade'] = df['T1']
df = df.drop(['T1'], axis=1)
st.write('df',df)