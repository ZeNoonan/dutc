# sourcery skip: merge-list-append
import pandas as pd
import numpy as np
from numpy import unique,where
import streamlit as st
from pandas import DataFrame,MultiIndex, Series,concat, get_dummies, NA, Timestamp, to_timedelta, concat as pd_concat, CategoricalIndex,date_range
from string import ascii_lowercase
from collections import Counter
from itertools import islice, groupby
from numpy.random import default_rng

st.set_page_config(layout="wide")

# youtube
#

def f():
    st.write("step 0")
    st.write("step 1")
    st.write("step 2")
    st.write("step 3")

f()

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
def f(data, *, mode=True):
    rv=[]
    for x in data:
        rv.append(x*2 if mode else x**2)
    return rv

st.write(f'{f([0,1,2,3],mode=False) = }')
    
### generator
def g(data,*,mode=True):
    for x in data:
        yield x*2 if mode else x**2

gi=g([0,1,2,3])
for x in gi:
    st.write(f'{x = }')

### generator coroutine ####

def c(mode=True):
    x = 0
    while True:
        x = yield x*2 if mode else x**2

ci = c() ; next(ci)
for x in [0,1,2,3]:
    st.write(f'{ci.send(x) = }')


xs=[1,2,3] # "strictly" homogenous
for x in xs:
    st.write(f'{x + 1 =}')

xs=[1,2.0,3+4j] # "loosely" homogenous (ie can treat them as though they were) integer float and complex number
for x in xs:
    st.write(f'{x + 1 =}')

xs=[1,2.0,'three'] # heterogenous
for x in xs:
    st.write(f'{x = }')

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
          ])).sort_index()
)

st.write('trades',trades)
st.write('trades unstack',trades.unstack('asset',fill_value=0))
st.write('trades update',trades
         .unstack('asset',fill_value=0)
         .groupby(lambda x: x == 'USD', axis='columns').sum()
         .set_axis(['volume','cash'],axis='columns')
         .join(trades.reset_index('asset')['asset'].loc[lambda s: s != 'USD'])
         .set_index('asset',append=True).sort_index()
         )


st.write('assets',assets)
st.write('prices',prices)







# https://www.dontusethiscode.com/blog/2024-07-10_ineq_joins.html

from pandas import DataFrame, Timedelta

state_polling_df = DataFrame(
    columns=[      'timestamp', 'device_id',  'state'],
    data=[
        ['2000-01-01 04:00:00',       'abc', 'state1'],
        ['2000-01-01 04:30:00',       'abc', 'state1'],
        ['2000-01-01 05:00:00',       'abc', 'state1'],
        ['2000-01-01 05:30:00',       'abc', 'state3'],
        ['2000-01-01 06:00:00',       'abc', 'state3'],
        ['2000-01-01 06:30:00',       'abc', 'state2'],
        ['2000-01-01 07:00:00',       'abc', 'state2'],
        ['2000-01-01 07:30:00',       'abc', 'state1'],

        ['2000-01-01 04:00:00',       'def', 'state2'],
        ['2000-01-01 04:30:00',       'def', 'state1'],
        ['2000-01-01 05:00:00',       'def', 'state3'],
        ['2000-01-01 05:30:00',       'def', 'state3'],
        ['2000-01-01 06:00:00',       'def', 'state1'],
        ['2000-01-01 06:30:00',       'def', 'state1'],
    ]
).astype({'timestamp': 'datetime64[ns]'})

alert_events = DataFrame(
    columns=[      'timestamp', 'device_id'],
    data=[
        ['2000-01-01 03:15:00',       'abc'],
        ['2000-01-01 04:05:00',       'abc'],
        ['2000-01-01 04:17:00',       'abc'],
        ['2000-01-01 04:44:00',       'abc'],
        ['2000-01-01 05:10:00',       'abc'],
        ['2000-01-01 05:23:00',       'abc'],
        ['2000-01-01 05:43:00',       'abc'],
        ['2000-01-01 05:53:00',       'abc'],
        ['2000-01-01 06:02:00',       'abc'],
        ['2000-01-01 06:08:00',       'abc'],
        ['2000-01-01 06:10:00',       'abc'],
        ['2000-01-01 06:23:00',       'abc'],
        ['2000-01-01 06:51:00',       'abc'],

        ['2000-01-01 03:05:00',       'def'],
        ['2000-01-01 04:15:00',       'def'],
        ['2000-01-01 04:27:00',       'def'],
        ['2000-01-01 04:34:00',       'def'],
        ['2000-01-01 05:20:00',       'def'],
        ['2000-01-01 05:33:00',       'def'],
        ['2000-01-01 06:22:00',       'def'],
        ['2000-01-01 06:29:00',       'def'],
        ['2000-01-01 06:43:00',       'def'],
        ['2000-01-01 07:01:00',       'def'],
    ]
).astype({'timestamp': 'datetime64[ns]'})

st.write(state_polling_df.head(), alert_events.head())
# state_polling_df.head(3)

state_df = (
    state_polling_df
    .assign(
        state_shift=lambda d:
            d.groupby('device_id')['state'].shift() != d['state'],
        state_group=lambda d: d.groupby('device_id')['state_shift'].cumsum(),
    )
    .groupby(['device_id', 'state', 'state_group'], as_index=False)
    .agg(
        start=('timestamp', 'min'),
        stop= ('timestamp', 'max'),
    )
)

st.write(state_df.sort_values(['device_id', 'start']))






# https://stackoverflow.com/questions/78692255/merge-dataframe-based-on-substring-column-labeld-while-keep-the-original-columns/78693761#78693761

df=pd.DataFrame({
    "[RATE] BOJ presser/2024-03-19T07:30:00Z/2024-03-19T10:30:00Z": [1],
    "[RATE] BOJ/2024-01-23T04:00:00Z/2024-01-23T07:00:00Z": [2],
    "[RATE] BOJ/2024-03-19T04:00:00Z/2024-03-19T07:00:00Z": [3],
    "[RATE] BOJ/2024-04-26T03:00:00Z/2024-04-26T06:00:00Z": [4],
    "[RATE] BOJ/2024-04-26T03:00:00Z/2024-04-26T08:00:00Z": [5],
    "[RATE] BOJ/2024-06-14T03:00:00Z/2024-06-14T06:00:00Z": [6],
    "[RATE] BOJ/2024-06-14T03:00:00Z/2024-06-14T08:00:00Z": [7],
    "[RATE] BOJ/2024-07-31T03:00:00Z/2024-07-31T06:00:00Z": [8],
    "[RATE] BOJ/2024-07-31T03:00:00Z/2024-07-31T08:00:00Z": [9],
    "[RATE] BOJ/2024-09-20T03:00:00Z/2024-09-20T06:00:00Z": [10],
    "[RATE] BOJ/2024-09-20T03:00:00Z/2024-09-20T08:00:00Z": [11],
    "[RATE] BOJ/2024-10-31T04:00:00Z/2024-10-31T07:00:00Z": [12],
    "[RATE] BOJ/2024-10-31T04:00:00Z/2024-10-31T09:00:00Z": [13],
    "[RATE] BOJ/2024-12-19T04:00:00Z/2024-12-19T07:00:00Z": [14],
    "[RATE] BOJ/2024-12-19T04:00:00Z/2024-12-19T09:00:00Z": [15],
})

st.write('dataframe', df.T)

result_want=pd.DataFrame({
        "[RATE] BOJ presser/2024-03-19T07:30:00Z/2024-03-19T10:30:00Z": [1],
        "[RATE] BOJ/2024-01-23T04:00:00Z/2024-01-23T07:00:00Z": [2],
        "[RATE] BOJ/2024-03-19T04:00:00Z/2024-03-19T07:00:00Z": [3],
        "[RATE] BOJ/2024-04-26T03:00:00Z/2024-04-26T06:00:00Z": [9],
        "[RATE] BOJ/2024-06-14T03:00:00Z/2024-06-14T06:00:00Z": [13],
        "[RATE] BOJ/2024-07-31T03:00:00Z/2024-07-31T06:00:00Z": [17],
        "[RATE] BOJ/2024-09-20T03:00:00Z/2024-09-20T06:00:00Z": [21]
    })

st.write('result',result_want.T)

groupings = (
    df.columns.str.extract(r'([^/]+)/(\d{4}-\d{2}-\d{2})')
)
unique = groupings.assign(orig=df.columns).drop_duplicates([0, 1])

result = (
    df.T
    .groupby([col.values for _, col in groupings.items()]).sum()
    .set_axis(unique['orig']).T
)

st.write('first solution',result.T)

def extract_unique(column):
    splitted = column.split('/')
    return splitted[0], splitted[1][:10]

result = {}
for _, col_group in groupby(sorted(df.columns), key=extract_unique):
    first, *remaining = col_group
    result[first] = df[[first, *remaining]].sum(axis=1)

result = pd.DataFrame(result)
st.write('2nd solution',result.T)










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











# https://www.dontusethiscode.com/blog/2024-06-12_groupby-sets.html

rng = default_rng(0)

center_ids = [f"Center_{i}" for i in range(2)]
locations = ["East", "West"]
service_types = ["Web Hosting", "Data Processing", "Cloud Storage"]

pd_df = DataFrame({
    "center_id":    rng.choice(center_ids, size=(size := 100)),
    "location":     rng.choice(locations, size=size),
    "service_type": rng.choice(service_types, size=size),
    "timestamp":    Timestamp.now() - to_timedelta(rng.integers(0, 3_650, size=size), unit='D'),
    "cpu_usage":    rng.uniform(0, 100, size=size),
    "mem_usage":    rng.uniform(0, 64, size=size),
})

st.write('pd df', pd_df)

results = []
aggfuncs = {'cpu_usage': 'mean', 'mem_usage': 'mean'}
groupings = [
    ['center_id', 'location', 'service_type'],
    ['service_type'],
    [],
]

for gs in groupings:
    if not gs:
        res = pd_df.agg(aggfuncs).to_frame().T
    else:
        res = pd_df.groupby(gs, as_index=False).agg(aggfuncs)
    results.append(res)
    
st.write('result',pd_concat(results, ignore_index=True))
# for gs in groupings:
#     st.write('this is gs within for loop',gs)
# st.write('this is res within for loop',pd_df.groupby('center_id', as_index=False).agg(aggfuncs))

test_results=[]
test_results.append(pd_df.groupby('service_type', as_index=False).agg(aggfuncs))
st.write('1 test results service type', test_results)
test_results.append(pd_df.groupby(['center_id', 'location', 'service_type'], as_index=False).agg(aggfuncs))
st.write('2 test results x 3 types', test_results)
# test_results.append(pd_df.groupby([]], as_index=False).agg(aggfuncs))
# st.write('3 test results', test_results)
# test_results.append(pd_df.groupby('service_type', as_index=False).agg(aggfuncs))
# st.write('4 test results', test_results)


def rollup(*items):
    "reversed version of itertools powerset recipe"
    return (
        [*islice(items, i, None)] for i in range(len(items)+1)
    )


st.write('rollup',rollup('center_id', 'location', 'service_type'))

# for gs in rollup('center_id', 'location', 'service_type'):
#     st.write(gs)


from pandas import concat, Series as pd_Series

results = []
aggfuncs = {'cpu_usage': 'mean', 'mem_usage': 'mean'}
for gs in rollup('center_id', 'location', 'service_type'):
    if not gs:
        res = pd_df.agg(aggfuncs).to_frame().T
    else:
        res = pd_df.groupby(list(gs), as_index=False).agg(aggfuncs)
    results.append(
        res.assign(
            groupings=lambda d: pd_Series([gs] * len(d), index=d.index)
        )
    )
    
st.write('result of the rollup',concat(results, ignore_index=True))

from itertools import chain, combinations

def cube(*items):
    """reversed version of itertools powerset recipe"""
    for size in range(len(items)+1, -1, -1):
        for combo in combinations(items, size):
            yield [*combo]

for gs in cube('center_id', 'location', 'service_type'):
    st.write(gs)

from pandas import concat, Series as pd_Series

results = []
aggfuncs = {'cpu_usage': 'mean', 'mem_usage': 'mean'}
for gs in cube('center_id', 'location', 'service_type'):
    if not gs:
        res = pd_df.agg(aggfuncs).to_frame().T
    else:
        res = pd_df.groupby(list(gs), as_index=False).agg(aggfuncs)
    results.append(
        res.assign(
            groupings=lambda d: pd_Series([gs] * len(d), index=d.index)
        )
    )
    
st.write(concat(results, ignore_index=True).fillna('')) # fillna to view output easily







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



# https://www.dontusethiscode.com/blog/2024-03-06_indexes_and_sets.html

from pandas import Series, date_range
from numpy.random import default_rng

rng = default_rng(0)
all_dates = date_range('2000-01-01', '2000-12-31', freq='D', name='date')

df = (
    DataFrame(
        index=all_dates,
        data={
            'temperature': rng.normal(60, scale=5, size=len(all_dates)),
            'precip': rng.uniform(0, 3, size=len(all_dates)), 
        }
    )
    .sample(frac=.99, random_state=rng) # sample 99% of our dataset
    .sort_index()
)

st.write('df',df)
st.write('days missing',
    df.reset_index()
    .merge(
        all_dates.to_frame(index=False), on='date', indicator=True, how='outer'
    )
    .query('_merge == "right_only"')
    ['date']
)

st.write('days missing full merge',
    df.reset_index()
    .merge(
        all_dates.to_frame(index=False), on='date', indicator=True, how='outer'
    ))

st.write('anser',all_dates.difference(df.index))

st.write('below is stackoverflow answer')
# https://stackoverflow.com/questions/77888782/wide-to-long-amid-merge/77888949#77888949

df1=pd.DataFrame({'company name':['A','B','C'],
               'analyst 1 name':['Tom','Mike',np.nan],
               'analyst 2 name':[np.nan,'Alice',np.nan],
               'analyst 3 name':['Jane','Steve','Alex']})

df2=pd.DataFrame({'company name':['A','B','C'],
               'score 1':[3,5,np.nan],
               'score 2':[np.nan,1,np.nan],
               'score 3':[6,np.nan,11]})

df_desire=pd.DataFrame({'company name':['A','A','B','B','B','C'],
               'analyst':['Tom','Jane','Mike','Alice','Steve','Alex'],
               'score':[3,6,5,1,np.nan,11]})

st.write('df1',df1)
st.write('df2',df2)
st.write('df desire',df_desire)
# st.write('df2 answer from different person')

# st.write('first questioner solution',pd.concat([df1.melt(id_vars='company name',value_vars=['analyst 1 name','analyst 2 name','analyst 3 name']),\
#            df2.melt(id_vars='company name',value_vars=['score 1','score 2','score 3'])],axis=1))

company_order = df1['company name'] # ensure all operations align correctly
frames = {'name': df1, 'score': df2}
frames = {
    name: d.set_index('company name').reindex(company_order)
    for name, d in frames.items()
}
st.write('frames',frames)

arr = np.dstack([*frames.values()]) # join all dataframes into 3d array
st.write('arr', arr)
n_companies, n_analysts, n_frames = arr.shape
result = (
    pd.DataFrame( # massage data into desired shape
        data=arr.reshape(-1, 2),
        index=company_order.repeat(n_analysts),
        columns=['name', 'score'],
    )
    .dropna(how='all')
)

st.write(result.reset_index())








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