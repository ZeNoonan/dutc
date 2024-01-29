import pandas as pd
import numpy as np
import streamlit as st
from pandas import DataFrame, Series, get_dummies, NA
from string import ascii_lowercase
from collections import Counter

st.set_page_config(layout="wide")

st.write('below comes from Blog Post on Pandas and Python Jan 26 2024')

from pandas import read_csv, DataFrame
from io import StringIO
from textwrap import dedent

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

st.write('below comes from stack overflow answer')


df = pd.DataFrame({
    'id':[1,2,3,1, 1],
    'time_stamp_date':['12','12', '12', '14', '14'],
    'sth':['col1','col1', 'col2','col2', 'col3']
})

st.write(df)

"""
 id  time_stamp_date sth
 1   12              col1
 2   12              col1
 3   12              col2
 1   14              col2
 1   14              col3
"""


"""
                    col1  col2  col3  col4
id time_stamp_date                        
1  12                  1     0     0     0
2  12                  1     0     0     0
3  12                  0     1     0     0
1  14                  0     1     1     0
"""


df.groupby(['id', 'time_stamp_date'], group_keys=True)['sth'].apply(get_dummies).fillna(0)


from pandas import Series, MultiIndex

s = Series(1, MultiIndex.from_frame(df))
s
sparse_s = s.unstack('sth', fill_value=0)
sparse_s

sth_list = ['col1', 'col2', 'col3', 'col4']
out = sparse_s.reindex(
    columns=sth_list,
    index=df[['id', 'time_stamp_date']].drop_duplicates(),
    fill_value=0
)

out



st.write('below comes from the blog post on new years resolutions')
# https://www.dontusethiscode.com/blog/2024-01-03_newyears_resolutions.html

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

st.write('below comes from stack overflow answer')
# https://stackoverflow.com/questions/77781290/filter-values-within-a-percentage-range-based-on-the-last-non-filtered-value/77781874#77781874


df = pd.DataFrame(
    {'trade': [100, NA, NA, 101, NA, 102, NA, 98, 107, NA, 101, NA, 98, NA, NA, 94]}
).astype({'trade': 'Int32'})

st.write('DF',df)

tmp = df.dropna()
st.write('after dropna', tmp)
valid = [0]
st.write('tmp.index[-1]',)
while valid[-1] < tmp.index[-1]:
    st.write('valid[-1]',valid[-1])
    st.write('tmp.index[-1]',tmp.index[-1])
    st.write("chunk:  =  tmp.loc[valid[-1]:, 'trade']",tmp.loc[valid[-1]:, 'trade'])
    chunk = tmp.loc[valid[-1]:, 'trade'] # get window of all unprocessed data
    st.write('chunk.iat[0]',chunk.iat[0])
    target = chunk.iat[0]
    valid.append(                        # find the first boundary
        chunk.between(target * .95, target * 1.05).idxmin() 
    )



st.write(
    f'{valid = }',  # [0, 8, 10, 15] (while loop took len(valid) iterations)
    df.assign(      # mask over values not in `valid`
        cleaned=lambda d: d['trade'].where(d.index.isin(valid)),
    ),
    sep='\n\n',
)

df=df.assign(cleaned=lambda d: d['trade'].where(d.index.isin(valid)))
st.write(df)
# valid = [0, 8, 10, 15]
#
#     trade  cleaned
# 0     100      100
# 1    <NA>     <NA>
# 2    <NA>     <NA>
# 3     101     <NA>
# 4    <NA>     <NA>
# 5     102     <NA>
# 6    <NA>     <NA>
# 7      98     <NA>
# 8     107      107
# 9    <NA>     <NA>
# 10    101      101
# 11   <NA>     <NA>
# 12     98     <NA>
# 13   <NA>     <NA>
# 14   <NA>     <NA>
# 15     94       94


# questioners answer interesting
stop = 0.05
# Prepare
df = pd.DataFrame(
    {'Trade': [100, NA, NA, 101, NA, 102, NA, 98, 107, NA, 101, NA, 98, NA, NA, 94]}
).astype({'Trade': 'Int32'})
trade = df['Trade']
trade.fillna(0, inplace=True)
trade = trade[trade != 0]

# Filter
trade.loc[(trade > trade.shift(1) * (1-stop)) & 
 (trade < trade.shift(1) * (1+stop))] = 0

# Recompose Dataframe
df = df.merge(trade.rename('T1'), left_index=True, 
right_index=True, how='left')
df['Trade'] = df['T1']
df = df.drop(['T1'], axis=1)
st.write('df',df)