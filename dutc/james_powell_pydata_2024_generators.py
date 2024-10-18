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

# youtube
# https://www.youtube.com/watch?v=Knth0LhQnC0&t=1011s

def f():
    st.write("step 0")
    st.write("step 1")
    st.write("step 2")
    st.write("step 3")

# f()

def g():
    st.write("step 0")
    yield
    st.write("step 1")
    yield
    st.write("step 2")
    yield
    st.write("step 3")

# gi=g()
# next(gi)
# # do some computation in here
# next(gi)
# #  do another computation in here
# next(gi)
# next(gi)

gi=g()
while True:
    try:
        x=next(gi)
    except StopIteration:
        break

# function/subroutine
def f(data, *, mode=True):  # sourcery skip: for-append-to-extend, list-comprehension
    rv=[]
    for x in data:
        rv.append(x*2 if mode else x**2)
    return rv

# st.write(f'{f([0,1,2,3],mode=False) = }')
    
### generator
def g(data,*,mode=True):
    for x in data:
        yield x*2 if mode else x**2

gi=g([0,1,2,3])
# for x in gi:
#     st.write(f'{x = }')

### generator coroutine ####

def c(mode=True):
    x = 0
    while True:
        x = yield x*2 if mode else x**2

ci = c() ; next(ci)
# for x in [0,1,2,3]:
#     st.write(f'{ci.send(x) = }')


xs=[1,2,3] # "strictly" homogenous
# for x in xs:
#     st.write(f'{x + 1 =}')

xs=[1,2.0,3+4j] # "loosely" homogenous (ie can treat them as though they were) integer float and complex number
# for x in xs:
#     st.write(f'{x + 1 =}')

xs=[1,2.0,'three'] # heterogenous
# for x in xs:
#     st.write(f'{x = }')

rng=default_rng(0)
dates=date_range('2020-01-01', periods=90)
assets=CategoricalIndex(unique(rng.choice([*ascii_lowercase],size=(10,4)).view('<U4').ravel()))

prices=(
    DataFrame(
        index=(idx := MultiIndex.from_product([
            dates,assets
    ], names='date asset'.split())),
    data={
        'buy': (
            rng.normal(
                loc=100,scale=20,size=len(assets)
            ).clip(0,500)
        *
            rng.normal(
                loc=1,scale=.01,size=(len(dates),len(assets))
            ).clip(.8,1.2).cumprod(axis=-1)
        ).ravel()
    }
    )
    .assign(
        sell=lambda df: df['buy'] * (1-abs(rng.normal(loc=0, scale=.02,size = len(df)))),
    )
    .round(2)
    )

# st.write('trying to fix trades')




trades = (
    Series(
        index=(idx := MultiIndex.from_product([
            dates,assets, range(10),
    ], names='date asset _'.split())),
    data=(
        rng.choice([+1,-1], size=len(idx))* rng.integers(100, 100_000, size=len(idx))
    ).round(-2)
    )
    .sample(random_state=rng, frac=.25)
    .sort_index()
    .pipe(lambda s: s
          .set_axis(
              MultiIndex.from_arrays([
                  s.index.get_level_values('date'),
                  range(len(s)),
                  s.index.get_level_values('asset'),
                  ],names='date trade# asset'.split())
          )
    )
    .pipe(lambda s:
          concat([
              s,
              Series(
                  index=MultiIndex.from_arrays([
                      s.index.get_level_values('date'),
                      s.index.get_level_values('trade#'),
                      ['USD'] * len(s),
                  ]),
                  data=(
                      prices
                      .loc[
                          MultiIndex.from_arrays([
                                s.index.get_level_values('date'),
                                s.index.get_level_values('asset'),  
                      ])
                    ]
                    .pipe(lambda df: where(s>0, df['buy'], df['sell']))
                    * -s.values
                  ),
              )
          ])
        )
        .sort_index()
)

# st.write('trades',trades)
# st.write('trades unstack',trades.unstack('asset',fill_value=0))
# st.write('trades update',trades
#          .unstack('asset',fill_value=0)
#          .groupby(lambda x: x == 'USD', axis='columns').sum()
#          .set_axis(['volume','cash'],axis='columns')
#          .join(trades.reset_index('asset')['asset'].loc[lambda s: s != 'USD'])
#          .set_index('asset',append=True).sort_index()
#          )

# my takeaway is that using stack and unstack allows Pandas to toggle between the two modes of homogenity
# st.write('assets',assets)
# st.write('prices',prices)


# raw_data=StringIO(dedent('''
#                   name,value
#                   abc,123
#                   def,456
#                   xyz,789
#                   ''').strip())

# data = {}
# for name,value in reader(raw_data):
#     data[name]=int(value)

with st.expander("example workings"):
    data={
        '2020Q1': 123,'2020Q2': 124,'2020Q3': 129,'2020Q4': 130,
        '2021Q1': 127,'2021Q2': 128,'2021Q3': 130,'2021Q4': 132,
    }

    prev_date, prev_value=None, None
    for curr_date,curr_value in data.items():
        if prev_date is None:
            prev_date, prev_value = curr_date, curr_value
            continue
        st.write(
            f'{prev_date} ~ {curr_date} {curr_value}',
            '\N{mathematical bold capital delta}QoQ',
            f'{prev_value - curr_value:>3}',
        )
        prev_date, prev_value = curr_date, curr_value


    st.write( "alternative way of writing it")
    for idx in range(1, len(data)):
        prev_date, prev_value = [*data.items()][idx-1]
        curr_date, curr_value = [*data.items()][idx]
        st.write(
            f'{prev_date} ~ {curr_date} {curr_value}',
            '\N{mathematical bold capital delta}QoQ',
            f'{prev_value - curr_value:>3}',

        )

    st.write('itertool solution')
    st.write('if you are a pandas user its basically a .shift opertion')
    for (prev_date, prev_value), (curr_date,curr_value) in pairwise(data.items()):
        st.write(
            f'{prev_date} ~ {curr_date} {curr_value}',
            '\N{mathematical bold capital delta}QoQ',
            f'{prev_value - curr_value:>3}',

        )
        
with st.expander('Start of NWISE function'):
    nwise = lambda g, *, n=2: zip(*(islice(g,i,None) for i, g in enumerate(tee(g,n))))
    nwise_longest = lambda g, *, n=2, fv=object(): zip_longest(
        *(islice(g,i,None) for i,g in enumerate(tee(g,n))), fillvalue=fv
    )

    first = lambda g, *, n=1: zip(chain(repeat(True,n), repeat(False)),g)
    last = lambda g, *, m=1, s=object(): ((y[-1] is s, x) for x,*y in nwise_longest(g,n=m+1,fv=s))

    for is_first, (is_last, x) in first(last(nwise(data.items()))):
        (prev_date, prev_value), (curr_date, curr_value) = x
        if is_first:
            st.write(f'{prev_date}   {"":>6}  {prev_value}')
        st.write(
            f'{prev_date} ~ {curr_date} {curr_value}',
            '\N{mathematical bold capital delta}QoQ',
            f'{prev_value - curr_value:>3}',
        )
        if is_last:
            st.write(f'{"":>6}   {curr_date}     {curr_value}')


    st.write('New')
    xs = 'abcdef'
    for idx in range(1,len(xs)):
        prev_x, curr_x = xs[idx-1], xs[idx]
        st.write(f'{prev_x, curr_x = }')

    for prev_x,curr_x in pairwise(xs):
        st.write(f'{prev_x, curr_x = }')


    strategies={
        'any':          'play any card',
        'hold-wilds':   'hold onto wild cards',
    }

    players = {
        'alice':        'any',
        'bob':          'hold-wilds',

    }

    # for pl in cycle(players):
    #     ...
    #     if players[pl] == 'any':
    #         candidates = {...}

