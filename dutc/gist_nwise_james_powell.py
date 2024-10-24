# sourcery skip: merge-list-append
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


# https://gist.github.com/dutc/4e2cdf42469548094ed22090b6634571

nwise = lambda g, *, n=2: zip(*(islice(g, i, None) for i, g in enumerate(tee(g, n))))
nwise_longest = lambda g, *, n=2, fv=object(): zip_longest(*(islice(g, i, None) for i, g in enumerate(tee(g, n))), fillvalue=fv)
first = lambda g, *, n=1: zip(chain(repeat(True, n), repeat(False)), g)
last = lambda g, *, m=1, s=object(): ((xs[-1] is s, x) for x, *xs in nwise_longest(g, n=m+1, fv=s))

if __name__ == '__main__':
  st.write(
    f"{[*nwise('abcd')] = }",
    f"{[*nwise_longest('abcd')] = }",
    f"{[*first('abcd')] = }",
    f"{[*last('abcd')] = }",
    sep='\n',
  )




