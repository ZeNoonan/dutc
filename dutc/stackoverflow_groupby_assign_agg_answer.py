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

with st.expander("groupby cam riddell answer"):
    st.write('')
    # https://stackoverflow.com/questions/78789249/how-to-use-python-pandas-groupby-for-multiple-columns/78789267#78789267
    buffer = StringIO('''
    Type    Product    Late or On Time
    A       X              On Time
    B       Y              Late
    C       Y              On Time
    C       X              On Time
    C       X              Late
    B       X              Late
    A       Y              Late
    C       Y              On Time
    B       Y              Late
    B       X              On Time
    ''')

    df = pd.read_table(buffer, sep='\s{2,}', engine='python')
    st.write(df)
    # df['late late']=df.where(df['Late or On Time']== 'Late') 
    # st.write(df)
    # df=df.assign(
    #     _late=df.where['Late or On Time'] == 'Late',
    #     _ontime=df.where['Late or On Time'] == 'On Time',
    # )
    df=df.assign(
        _late=lambda d: d['Late or On Time'] == 'Late',
        _ontime=lambda d: d['Late or On Time'] == 'On Time',
    )
    st.write('assign',df)
    # st.write  ( df.groupby(['Type','Product'])['Product'].count()  )

    st.write(
    df.assign(
        _late=lambda d: d['Late or On Time'] == 'Late',
        _ontime=lambda d: d['Late or On Time'] == 'On Time',
    )
    .groupby(['Type', 'Product'], as_index=False)
    .agg(**{
        'Product/Type Count': ('Product', 'size'),
        'Late Count':         ('_late', 'sum'),
        'On Time Count':      ('_ontime', 'sum'),
    })
)