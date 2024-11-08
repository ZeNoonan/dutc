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
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(layout="wide")

from textwrap import dedent
from io import StringIO

buffer = StringIO(dedent('''
    Item	N1	N2	N3	N4
    Item1	1	2	4	8
    Item2	2	3	6	7
    Item3	4	5	7	9
    Item4	1	5	6	7
    Item5	3	4	7	8
'''))

from pandas import read_table, crosstab

df = read_table(buffer)
melted = df.melt('Item')
st.write(melted.head())

st.write(
    crosstab(melted['Item'], melted['value'])
    .rename_axis(index=None, columns=None)
)

st.write(melted.value_counts(['Item', 'value']).sort_index().head(8))

st.write(
    melted.astype({'Item': 'category', 'value': 'category'})
    .groupby(['Item', 'value'], observed=False)
    .size()
).head(6) # the full output is quite long

result = melted.value_counts(['Item', 'value'])
st.write(result.head(8))

from pandas import MultiIndex

st.write(result.reindex(
    MultiIndex.from_product(result.index.levels),
    fill_value=0,
).head(8))

