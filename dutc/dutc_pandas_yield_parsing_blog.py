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

st.write('below comes from Blog Post on Pandas and Python Jan 26 2024')
# https://www.dontusethiscode.com/blog/2024-01-24_pandas_needs_python.html

buffer = StringIO(dedent('''
    device,upgrade_dates
    device-1,2000-01-01,2000-02-01,2000-03-01
    device-2,2000-01-01,2000-04-01
    device-3,2000-01-01,2000-03-01,2000-05-01,2000-10-01
    device-4,2000-01-01,2000-07-01,2000-09-01
''').strip())

st.write('buffer read:',buffer.read())
st.write('buffer', buffer)

from traceback import print_exc
from contextlib import contextmanager

# @contextmanager
# def short_traceback(limit=1):
#     try:
#         yield
#     except Exception:
#         print_exc(limit=limit)

# buffer.seek(0)
# with short_traceback(1):
#     read_csv(buffer)

buffer.seek(0)
st.write('read csv',read_csv(buffer, on_bad_lines='skip'))

def process(f):
    f = (ln.strip() for ln in f)
    yield next(f).split(',')
    for line in f:
        dev, *dates = line.split(',')
        yield dev, dates
        
buffer.seek(0)
head, *body = process(buffer)
df = DataFrame(body, columns=head)
st.write('df',df)

st.write(df.dtypes)

st.write('explode df',df.explode('upgrade_dates'))

def process(f):
    f = (ln.strip() for ln in f)
    yield next(f).split(',')
    for line in f:
        dev, *dates = line.split(',')
        for d in dates: # explode as pre-processing step
            yield dev, d

buffer.seek(0)
head, *body = process(buffer)
st.write(DataFrame(body, columns=head))