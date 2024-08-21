# sourcery skip: merge-list-append
import pandas as pd
import numpy as np
from numpy import unique,where,array,tile
import streamlit as st
from pandas import DataFrame,MultiIndex,period_range,IndexSlice, Series,merge,concat, get_dummies, NA, Timestamp, to_timedelta, concat as pd_concat, CategoricalIndex,date_range
from string import ascii_lowercase
from collections import Counter
from itertools import islice, groupby, pairwise, cycle, tee, zip_longest, chain, repeat, takewhile
from numpy.random import default_rng
from io import StringIO
from textwrap import dedent
from csv import reader
from collections import deque

st.set_page_config(layout="wide")

with st.expander("James Powell Pandas"):
    # https://www.youtube.com/watch?v=J8dJAekOkSU
    monthly_sales = DataFrame({
    'store': ['New York', 'New York', 'New York','San Francisco','San Francisco','San Francisco','Chicago','Chicago','Chicago'],
    'item': [ 'milkshake' ,  'french fry' ,  'hamburger','milkshake' ,  'french fry' ,  'hamburger','milkshake' ,  'french fry' ,  'hamburger' ],
    'quantity': [ 100,110,120,200,210,220,300,310,320 ],
    }).set_index(['store','item'])
    forecast = DataFrame({
    'quarter': ['2025Q1', '2025Q2', '2025Q3', '2025Q4'],
    'value': [ 1.01 ,  1.02 ,  1.03 ,  1.04 ],
    }).set_index('quarter')
    st.write('sales', monthly_sales,'forecast',forecast)

    index = MultiIndex.from_product([*monthly_sales.index.levels,forecast.index])
    st.write('monthly sales level',monthly_sales.index.levels)
    st.write('monthly sales index',monthly_sales.index)
    st.write('index',index)
    with st.echo():
        st.write('reindex',forecast.reindex(index,level=forecast.index.name))
        # st.write('level name',forecast.index.name)
        st.write('monthly sales', monthly_sales)
        st.write(monthly_sales * forecast.reindex(index,level=forecast.index.name))

    rng=default_rng(0)
    df = DataFrame(
        index=(idx := MultiIndex.from_product([
            period_range('2000-01-01', periods=8,freq='Q'),
            ['New York','Chicago','San Francisco'],
        ],names=['quarter','location'])),
        data={
            'revenue': +rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
            'expenses': -rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
            'manager': tile(['Alice Adams','Bob Brown','Charlie Cook','Dana Daniels'], len(idx)//4),
        }
    ).pipe(lambda df:
           merge(
               df,
               Series(
                   index=(idx := df['manager'].unique()),
                   data=rng.choice([True,False],p=[.75,.25],size=len(idx)),
                   name='franchise',
               ),
               left_on='manager',
               right_index=True,
        )
    )

    st.write('mad df',df.sort_index())
    st.write(df[~df['franchise']])

    # test_df = DataFrame(
    #     index=(idx := MultiIndex.from_product([
    #         period_range('2000-01-01', periods=8,freq='D'),
    #         ['New York','Chicago','San Francisco'],
    #     ],names=['quarter','location'])),
    #     data={
    #         'revenue': +rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
    #         'expenses': -rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
    #         'manager': tile(['Alice Adams','Bob Brown','Charlie Cook'], len(idx)//3),
    #     }
    # )

    sales = DataFrame(
        index=(idx := MultiIndex.from_product([
            date_range('2000-01-01', periods=20,freq='D'),
            ['New York','Chicago','San Francisco','Houston'],
            ['burger','fries','combo','milkshakes'],
        ],names=['date','location','item'])),
        data={
            'quantity': +rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
            
        }
    )
    

    sales_test = DataFrame(
        index=(idx := MultiIndex.from_product([
            date_range('2000-01-01', periods=20,freq='D'),
            ['New York','Chicago','San Francisco'],
            # tile(['burger','fries','combo','milkshakes'], len(idx)//4),
        ],names=['date','location'])),
        data={
            'quantity': +rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
            'item': tile(['burger','fries','combo','milkshakes'], len(idx)//4),
            
        }
    )
    # .set_index(['date','location','item'])
    # st.write('test sales', sales_test,'PUT in index')

    # st.write('sales',sales,'unstack sales',sales.unstack('item','fill_value=0'))
    # st.write('test sales',sales_test)


    # Define the MultiIndex
    date_rng = pd.date_range('2000-01-01', periods=20, freq='D')
    locations = ['New York', 'Chicago', 'San Francisco']
    items = ['hamburger', 'french fry', 'combo', 'milkshake']

    idx = pd.MultiIndex.from_product([date_rng, locations, items], names=['date', 'location', 'item'])

    # Generate random sales data
    np.random.seed(42)
    quantities = np.random.normal(loc=10_000, scale=1_000, size=len(idx)).round(-1)

    # Create the DataFrame
    sales = pd.DataFrame({'quantity': quantities}, index=idx)

    # Randomly select some 'combo' rows to set their quantity to zero
    combo_idx = sales.index.get_level_values('item') == 'combo'
    combo_rows = sales[combo_idx]

    # Choose a random subset of these combo rows
    random_combo_idx = np.random.choice(combo_rows.index, size=int(len(combo_rows) * 0.3), replace=False)

    # Set the 'quantity' of the selected random combo rows to zero
    sales.loc[random_combo_idx, 'quantity'] = 0

    # Drop some random dates to simulate missing dates
    dates_to_remove = np.random.choice(date_rng, size=3, replace=False)
    df = sales.drop(index=dates_to_remove, level='date')

    # Display the result
    st.write('update from chat gpt where we have some combos=0 and some missing dates',df,'unstack on this',df.unstack('item','fill_value=0'))


    
    unstack_sales=(df
             .unstack('item','fill_value=0')
             .pipe(lambda df: df
                   .reindex(
                       MultiIndex.from_product([
                           date_range((dts := df.index.get_level_values('date')).min(), dts.max(), freq='D'),
                           df.index.get_level_values('location').unique(),
                        ], names=df.index.names),
                        fill_value=0,
                   )
                   ).pipe(lambda df: df.droplevel(0, axis=1))
             )
    # st.write('this is the clean up operations',unstack_sales)
    # unstack_sales.columns = unstack_sales.columns.droplevel(0) # for some reason when I do unstack i have to drop down a level for column names, james powell
    # didn't seem to get this error
    # fixed by asking chat gpt how i could fit the above into it so added this to james code
    # .pipe(lambda df: df.droplevel(0, axis=1))
    st.write(unstack_sales.loc[IndexSlice[:,['Chicago','New York'],   :]])
    st.write(unstack_sales.columns)

    st.write(df
             .unstack('item','fill_value=0')
             .pipe(lambda df: df
                   .reindex(
                       MultiIndex.from_product([
                           date_range((dts := df.index.get_level_values('date')).min(), dts.max(), freq='D'),
                           df.index.get_level_values('location').unique(),
                        ], names=df.index.names),
                        fill_value=0,
                   )
                   ).pipe(lambda df: df.droplevel(0, axis=1)).pipe(lambda df:
                          concat([
                              concat([
                                  df.loc[IndexSlice[:,['Chicago','New York'],   :]][['french fry','hamburger','milkshake',]],
                                  df.loc[IndexSlice[:,['San Francisco'],        :]][['french fry','hamburger',            ]].assign(milkshake=0),
                                  ],axis='index').sort_index()
                              .rename(lambda x: ('combo',x), axis='columns'),
                              df[['french fry','hamburger','milkshake']]
                              .rename(lambda x: ('solo', x), axis='columns'),
                          ],axis='columns')
                          .pipe(lambda df: df.set_axis(MultiIndex.from_tuples(df.columns),axis='columns'))
                          .rename_axis(['order type', 'item'], axis='columns')
            )
    
    
                          .sort_index()
    # )
                          .groupby('item',axis='columns').sum()
             )
                    
    # st.write('full output', full_output)
    # st.write('index length',len(idx := MultiIndex.from_product([
    #         period_range('2000-01-01', periods=8,freq='D'),
    #         ['New York','Chicago','San Francisco'],
    #     ],names=['quarter','location'])))
    # st.write('index',DataFrame(index=(idx := MultiIndex.from_product([
    #         period_range('2000-01-01', periods=8,freq='D'),
    #         ['New York','Chicago','San Francisco'],
    #     ],names=['quarter','location']))))
    
    # st.write('test df', test_df.sort_index())
    # st.write(len(idx))
    # st.write(len(idx)//7)
    # st.write(tile(['Alice Adams','Bob Brown','Charlie Cook','Dana Daniels'], len(idx)//4))
    # st.write(tile(['Alice Adams','Bob Brown','Charlie Cook','Dana Daniels'], len(idx)//12))

with st.expander("groupby cam riddell answer"):
    st.write('')
    # https://stackoverflow.com/questions/78789249/how-to-use-python-pandas-groupby-for-multiple-columns/78789267#78789267
    buffer = StringIO('''
    Type    Product    Late or On Time
    A       X              On Time
    B       Y              Late
    C       Y              On Time
    C       X              On Time
    C       X              Late
    B       X              Late
    A       Y              Late
    C       Y              On Time
    B       Y              Late
    B       X              On Time
    ''')

    df = pd.read_table(buffer, sep='\s{2,}', engine='python')
    st.write(df)
    # df['late late']=df.where(df['Late or On Time']== 'Late') 
    # st.write(df)
    # df=df.assign(
    #     _late=df.where['Late or On Time'] == 'Late',
    #     _ontime=df.where['Late or On Time'] == 'On Time',
    # )
    df=df.assign(
        _late=lambda d: d['Late or On Time'] == 'Late',
        _ontime=lambda d: d['Late or On Time'] == 'On Time',
    )
    st.write('assign',df)
    # st.write  ( df.groupby(['Type','Product'])['Product'].count()  )


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


# https://www.youtube.com/watch?v=iKzOBWOHGFE&t=946s
with st.expander("james powell newton method"):
    f=lambda x: (x+2) * (x-4)
    fprime=lambda x: 2*x - 2
    def newton(f, x0, fprime):
        x = x0
        for _ in range(50):
            x -= f(x) / fprime(x)
            return x

    def newton(f,x,fprime):
        while True:
            x -= f(x) / fprime(x)
            yield x

    # st.write([list((newton(f,10,fprime)))])
    st.write('islice below xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    st.write([list(islice(newton(f,10,fprime),5))])
    st.write('islice above xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    st.write('Deque below xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    st.write((deque(islice(newton(f,10,fprime),5),maxlen=1)[0]))
    st.write('Deque above xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # st.write(list(takewhile(lambda x: np.isclose(f(x),0,abs_tol=3), islice(newton(f,10,fprime),3))))
    values = list(takewhile(lambda x: np.isclose(f(x), 0, atol=3), islice(newton(f, 10, fprime), 5)))
    st.write('values',values)

    f = lambda x: (x + 2) * (x - 4)
    fprime = lambda x: 2 * x - 2

    def newton(f, x, fprime):
        while True:
            x -= f(x) / fprime(x)
            yield x

    # Assuming you want to process each element of the array separately
    results = [list(islice(newton(f, xi, fprime), 5)) for xi in np.array([-10, 10])]

    st.write(results)
    for xi in np.array([-10, 10]):
        st.write('array', xi)



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

# sourcery skip: merge-list-append
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