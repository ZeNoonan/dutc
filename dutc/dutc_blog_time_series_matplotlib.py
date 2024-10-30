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

# plt.style.use('fivethirtyeight')
# plt.style.use('classic')
plt.style.use('Solarize_Light2')
#https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html


with st.expander('dummy example'):
    from datetime import datetime, timedelta

    def custom():
        seen = []
        def _inner(date):
            if not seen or seen[-1].year != date.year:
                label = f"{date:%b\n%Y}"
            else:
                label = f"{date:%b}"
            seen.append(date)
            return label
        return _inner

    # Create a list of 10 dates
    start_date = datetime(2022, 11, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(10)]

    # Create our formatter
    formatter = custom() # DN note look function is nearly called in this instance, it only needs the brackets and away you go

    # Apply the formatter to each date
    formatted_dates = [formatter(date) for date in dates] # DN finally understand, function is properly called now as brackets is put in , DN wonder could i call it inner instead of formatter to make it explict

    # Print the results
    for original, formatted in zip(dates, formatted_dates):
        st.write(f"{original.strftime('%Y-%m-%d')}: {formatted}")




from pandas import DataFrame, date_range
from numpy.random import default_rng

rng = default_rng(0)

df = DataFrame(
    index=(idx := date_range('2000', '2004', freq='D')),
    data={
        'A': 10_000 + rng.normal(1, .01, size=idx.size).cumprod(),
        'B': 10_000 + rng.normal(1, .01, size=idx.size).cumprod(),
    }
)

st.write(df.head())
# st.write('stack', df.stack().reset_index())
altair_df=df.stack().reset_index().rename(columns={'level_1':'symbol','level_0':'date',0:'amount'})
st.write(altair_df)
st.altair_chart(alt.Chart(altair_df).mark_line().encode(
    x='date:T',
    y='amount:Q',
    color='symbol:N',
))

# %matplotlib agg

from matplotlib.pyplot import rc
rc('font', size=8)
rc('figure', facecolor='white', figsize=(10, 2))

from matplotlib.pyplot import subplots

fig, ax = subplots()

ax.plot(df.index, df['A'])
ax.plot(df.index, df['B'])

st.write('first attempt')
st.pyplot(fig)

from matplotlib.dates import AutoDateLocator, DateFormatter

locator = AutoDateLocator()
ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
st.write('2nd attempt',fig)

# fig, ax = subplots()
# ... some plotting code

# specify a locator & formatter for the a-axis
# ax.xaxis.set_major_locator(FixedLocator([1, 2, 3])) # set tick positions
# ax.xaxis.set_major_formatter(NullFormatter())       # remove tick labels

with st.echo():
    def outer():
        def inner():
            st.write("Inner function executed")
        return inner  # Returning the function object - this part i didnt get, had to run an example to understand that a function object is returned if you dont put brackets after it

    result = outer()  # 'result' is now the 'inner' function
    # No print output yet, because 'inner' hasn't been called

    result()  # Now we call 'inner', and see the print output


from matplotlib.dates import num2date

def custom():
    seen = []
    def _inner(value, pos):
        cur = num2date(value)
        # st.write('value',value,'numdate(value)',num2date(value))
        if not seen or seen[-1].year != cur.year:
            label = f'{cur:%b\n%Y}'
        else:
            label = f'{cur:%b}'
        seen.append(cur)
        return label
    return _inner

ax.xaxis.set_major_formatter(custom())
st.write('3rd attempt',fig)
st.pyplot(fig)

# Given the above closure pattern, we can accomplish the same result through the use of a Python Generator.
#  This is mainly for fun and doesn’t necessarily provide any benefits to your plotting code.

from matplotlib.dates import num2date

def custom():
    seen = []
    seen.append(num2date((yield)))
    while True:
        if len(seen) <= 1 or seen[-2].year != seen[-1].year:
            label = f'{seen[-1]:%b\n%Y}'
        else:
            label = f'{seen[-1]:%b}'
        seen.append(num2date((yield label)))

formatter = custom()
next(formatter)
ax.xaxis.set_major_formatter(lambda value, pos: formatter.send(value))
st.write('4th attempt with generator',fig)

with st.expander('testing the custom() generator on a pandas dataframe to help understanding'):
    def custom():
        seen = []
        seen.append((yield))
        while True:
            # Convert Timestamp to datetime if necessary
            current_date = seen[-1].to_pydatetime() if hasattr(seen[-1], 'to_pydatetime') else seen[-1]
            
            if len(seen) <= 1 or seen[-2].to_pydatetime().year != current_date.year:
                label = f'{current_date:%b\n%Y}'
            else:
                label = f'{current_date:%b}'
                
            seen.append((yield label))

    # Create a sample DataFrame with dates
    start_date = datetime(2022, 11, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(10)]
    df = pd.DataFrame({'Date': dates})

    # Initialize our formatter
    formatter = custom()
    next(formatter)  # Prime the generator

    # Apply the formatter to the DataFrame
    df['Formatted_Date'] = df['Date'].apply(lambda x: formatter.send(x))
    st.write(df)

with st.expander('why is there a yield in the first part'):
    with st.echo():
        # https://claude.ai/chat/59490da4-d931-459c-a071-6b0ee02e67c8
        def simplified_custom():
            seen = []
            # Prime the generator and get the first date
            # first_date = yield
            # seen.append(first_date)
            # DN testing the below
            # first_date = yield
            seen.append((yield)) # DN wow this worked, same as cam riddell example, my question why is the brackets around yield necessary

            
            while True:
                # Format and yield the label, then get the next date
                if len(seen) == 1 or seen[-1].year != seen[-2].year:
                    label = f"{seen[-1]:%b\n%Y}"
                else:
                    label = f"{seen[-1]:%b}"
                
                next_date = yield label
                seen.append(next_date)

        # Usage:
        formatter = simplified_custom()
        next(formatter)  # Advance to the first yield
        st.write(formatter.send(datetime(2023, 1, 1)))  # Send first date, get first label
        st.write(formatter.send(datetime(2023, 2, 1)))  # Send second date, get second label



from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

locator = AutoDateLocator()
formatter = ConciseDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

st.write(fig)

locator = AutoDateLocator()
formatter = ConciseDateFormatter(
    locator,
    formats=[    # default precision levels
        '%Y',    # years
        '%b',    # months
        '%d',    # days
        '%H:%M', # hours
        '%H:%M', # minutes
        '%S.%f', # seconds
    ],
)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

st.write(fig)

locator = AutoDateLocator()
formatter = ConciseDateFormatter(
    locator,
    formats=[     # update the format when we have precision at the Year level
        '%b\n%Y', # years (show month + year)
        '%b',     # months
        '%d',     # days
        '%H:%M',  # hours
        '%H:%M',  # minutes
        '%S.%f',  # seconds
    ],
)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

st.write(fig)