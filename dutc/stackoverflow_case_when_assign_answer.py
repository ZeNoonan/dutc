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


# https://stackoverflow.com/questions/78570861/smarter-way-to-create-diff-between-two-pandas-dataframes/78571153#78571153

dir_old = pd.DataFrame([
    {"Filepath": "dir1/file1", "Hash": "hash1"},
    {"Filepath": "dir1/file2", "Hash": "hash2"},
    {"Filepath": "dir2/file3", "Hash": "hash3"},
])

dir_new = pd.DataFrame([
    # {"Filepath": "dir1/file1", "Hash": "hash1"}, # deleted file
    {"Filepath": "dir1/file2", "Hash": "hash2"},
    {"Filepath": "dir2/file3", "Hash": "hash5"},  # changed file
    {"Filepath": "dir1/file4", "Hash": "hash4"},  # new file
])

st.write('dir old', dir_old, 'dir ;new', dir_new)
df_merged = pd.merge(dir_new, dir_old, how='outer', indicator=True)
st.write(df_merged)

st.write(
    df_merged
    .assign(
        Hash=lambda d: d['Hash_x'].fillna(d['Hash_y']),
        Status=lambda d:  # NA is our fallthrough value (if cases are not exhaustive)
            pd.Series(pd.NA, index=d.index, dtype='string')
            .case_when([
                (d['_merge'] == 'right_only',                          'deleted'  ),
                (d['_merge'] == 'left_only',                           'created'  ),
                (d['_merge'].eq('both') & d['Hash_x'].ne(d['Hash_y']), 'changed'  ),
                (d['_merge'].eq('both') & d['Hash_x'].eq(d['Hash_y']), 'unchanged'),
            ]),
    )
    .drop(columns=['Hash_x', 'Hash_y', '_merge'])
)
#      Filepath   Hash     Status
# 0  dir1/file1  hash1    deleted
# 1  dir1/file2  hash2  unchanged
# 2  dir1/file4  hash4    created
# 3  dir2/file3  hash5    changed