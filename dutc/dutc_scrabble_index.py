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

st.write('below comes from the blog post on new years resolutions')
# https://www.dontusethiscode.com/blog/2024-01-03_newyears_resolutions.html

# The Index is a huge feature of pandas, and, if you fight against it every day,
#  I suggest trying to work with it instead. In this Scrabble simulator I created, look at how simple the implementation of scoring words is:
#

points = {
    'A': 1, 'E': 1, 'I': 1, 'O': 1, 'U': 1,
    'L': 1, 'N': 1, 'S': 1, 'T': 1, 'R': 1,
    'D': 2, 'G': 2, 'B': 3, 'C': 3, 'M': 3,
    'P': 3, 'F': 4, 'H': 4, 'V': 4, 'W': 4,
    'Y': 4, 'K': 5, 'J': 8, 'X': 8, 'Q': 10, 'Z': 10
}
points = {k.lower(): v for k, v in points.items()}

words = ['hello', 'world', 'test', 'python', 'think']
words_df = (
    DataFrame.from_records([Counter(w) for w in words], index=words)
    .reindex([*ascii_lowercase], axis=1)
    .fillna(0)
    .astype(int)
)

words_df @ Series(points) # Compute the point value for all words in DataFrame

