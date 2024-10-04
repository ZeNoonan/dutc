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

st.write('below is stackoverflow answer')
# https://stackoverflow.com/questions/77888782/wide-to-long-amid-merge/77888949#77888949

df1=pd.DataFrame({'company name':['A','B','C'],
               'analyst 1 name':['Tom','Mike',np.nan],
               'analyst 2 name':[np.nan,'Alice',np.nan],
               'analyst 3 name':['Jane','Steve','Alex']})

df2=pd.DataFrame({'company name':['A','B','C'],
               'score 1':[3,5,np.nan],
               'score 2':[np.nan,1,np.nan],
               'score 3':[6,np.nan,11]})

df_desire=pd.DataFrame({'company name':['A','A','B','B','B','C'],
               'analyst':['Tom','Jane','Mike','Alice','Steve','Alex'],
               'score':[3,6,5,1,np.nan,11]})

st.write('df1',df1)
st.write('df2',df2)
st.write('df desire',df_desire)
# st.write('df2 answer from different person')

# st.write('first questioner solution',pd.concat([df1.melt(id_vars='company name',value_vars=['analyst 1 name','analyst 2 name','analyst 3 name']),\
#            df2.melt(id_vars='company name',value_vars=['score 1','score 2','score 3'])],axis=1))

company_order = df1['company name'] # ensure all operations align correctly
frames = {'name': df1, 'score': df2}
frames = {
    name: d.set_index('company name').reindex(company_order)
    for name, d in frames.items()
}
st.write('frames',frames)

arr = np.dstack([*frames.values()]) # join all dataframes into 3d array
st.write('arr', arr)
n_companies, n_analysts, n_frames = arr.shape
result = (
    pd.DataFrame( # massage data into desired shape
        data=arr.reshape(-1, 2),
        index=company_order.repeat(n_analysts),
        columns=['name', 'score'],
    )
    .dropna(how='all')
)

st.write(result.reset_index())

