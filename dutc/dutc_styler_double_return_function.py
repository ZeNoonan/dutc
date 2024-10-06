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

# https://www.dontusethiscode.com/blog/2024-04-17_joins.html

from pandas import DataFrame

df_left = DataFrame({
    'group': ['a', 'b', 'c', 'd'],
    'value': [ 1 ,  2 ,  3 ,  4 ],
})

df_right = DataFrame({
    'group': ['c', 'd', 'e', 'f' ],
    'value': [ -3,  -4,  -5,  -6 ],
})

s_left, s_right = set(df_left['group']), set(df_right['group'])

st.write(df_left,df_right,s_left,s_right)

from functools import partial

def color_parts(df):
    def _color_row(row):
        bg_dict = {'left_only': '#FFDAB9', 'right_only': '#ADD8E6'}
        style = ['']
        if row['_merge'] in bg_dict:
            style = [
                f"background-color: {bg_dict[row['_merge']]}; color: black"
            ]
        return style * len(row)
    return (
        df.style.apply(_color_row, axis=1)
        .format('{:g}', subset=df.select_dtypes('number').columns)
    )

# outer_join=df_left.merge(df_right, on='group', how='outer', indicator=True).pipe(color_parts),set_union=s_left.union(s_right)
