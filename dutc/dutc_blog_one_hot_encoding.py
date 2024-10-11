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

# https://www.dontusethiscode.com/blog/2024-09-25_dataframes_one_hot.html

# Let’s start by making some label data, with some known unique categories.

from pandas import Series, DataFrame

pd_dense = Series([*'ABCDA'], dtype='category', name='label')
pd_dense.to_frame()

#To one-hot encode this data, we can iterate over the unique categories from the data and simply create a new column for each label. 
# In each of these new columns, we observe a binary output where 1s are an affirmation that the given label appeared in this position and 0s indicate the opposite.

# Take a look at the following output and see if you can logically map where each label 'A' appeared in the previous pandas.DataFrame.

(
    DataFrame({
        label: pd_dense == label for label in pd_dense.cat.categories
    })
    .astype(int)
)

# While doing transformations in a mechanical manner (as we did above) is a fun learning exercise, we can also use the very 
# handy pandas.get_dummies function to arrive at the same result.

from pandas import get_dummies

pd_sparse = get_dummies(pd_dense).astype(int)
pd_sparse