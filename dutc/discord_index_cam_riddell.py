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

from numpy.random import default_rng
from pandas import Series, MultiIndex, date_range

# Just a quick qualifier — for all analyses working with data under a few gigabytes in size, pandas is still a very fine tool to use and will likely continue to be used for quite time.

# To your question, I don't think that we will see another index-oriented tabular DataFrame library. 
# I haven't see any spinoffs of an index-aware data structure (aside form XArray). And as convenient as the index is, 
# it proves to be a tricky concept for new users to make sense of and causes a lot of "wrestling" with pandas code.

# Definitely, while there is nothing intrinsically wrong with the index (or the idea of one) I do think that the Index is a topic that is still hardly 
# mentioned when teaching pandas pandas despite it having such a profound impact on analytical results. For example, users are often 
# surprised when their values are "magically" replaced with NaN when performing a column assignment.

rng = default_rng(0)

df = Series(
    index=MultiIndex.from_product(
        [['A', 'B'], date_range('2000-01', freq='D', periods=3)],
        names=['group', 'date'],
    ),
    data=rng.normal(50, 10, size=6),
    name='value'
).to_frame()
print(df)
#                       value
# group date
# A     2000-01-01  51.257302
#       2000-01-02  48.678951
#       2000-01-03  56.404227
# B     2000-01-01  51.049001
#       2000-01-02  44.643306
#       2000-01-03  53.615951

def demean_groups(data): # please Don't Use This Code, avoid UDF's!
    return data - data.mean()

result = df.groupby('group')['value'].transform(demean_groups).reset_index(drop=True)
print(result)
# 0   -0.856191
# 1   -3.434542
# 2    4.290733
# 3    1.279582
# 4   -5.126113
# 5    3.846531
# Name: value, dtype: float64

df['t'] = result # silent NaN fill from index-alignment
#                       value   t
# group date
# A     2000-01-01  51.257302 NaN
#       2000-01-02  48.678951 NaN
#       2000-01-03  56.404227 NaN
# B     2000-01-01  51.049001 NaN
#       2000-01-02  44.643306 NaN
#       2000-01-03  53.615951 NaN

# I have personally felt an odd disconnect between the importance of index-alignment in pandas operations and the support it has in the pandas library. 
# Standard single column indexes have had good, first-class citizen support for as long as I can remember. However only in the 5 or so years has 
# pandas really began making more flexible use of the MultiIndex. It used to be very clunky to work with but is now much more flexible.

# I myself have wondered of the origins of the Index idea and haven't done the archaeology to figure out how it really came about,
# but my guess tied pandas back into its origins of the finance industry. Datetime operations are incredibly common in this sector, 
# and so working well with datetimes was likely a priority. If you have two separate tables, both of which have a date column, 
# you would have to explicitly join these tables before performing any operations. But if each of these tables had a "smart" column (an .index) 
# then you could actually focus on the math at hand and let the data align itself. It is the difference between writing something like:

df1, df2 = ..., ...

# temporary intermediate for algebraic operation
intermediate = df1.merge(df2, on='date', suffixes=('_left', '_right'), how='left')
desired_result = intermediate['value_left'] + intermediate['value_right']

# assign intermediate back to original data for continued processing
df1['combined'] = intermediate

# Where as if you use an implicit merge (on an index), you could do...

df1, df2 = ....set_index('date'), ....set_index('date')
df1['combined'] = df1['value'] + df2['value']

# Which is much less verbose and easily readable as well- it lets you focus on the operations rather than the mechanics of the operation.

# pandas has seen many contributions over the last 15+ years of its development and I don't believe there was a strong structured view
#  of what the Index should be/do for long while. So over time many contributions were likely accepted that didn't fully agree with the
#  idea of an Index and once that begins it becomes harder for core developers to go back and retrofit Index principles to those additions. 
# This is just my guess as to why the Index has received spotty support in the early pandas days. That said, I know some core developers
#  are Index advocates and some are even opposed to it. The main argument is to create a product that has less "surprising" behavior, 
# and unfortunately better documentation is only part of the problem. As far as tabular data analysis tools are concerned, 
# pandas having an index is a unique idea- you don't see the Index used in the same way in R or SQL which have been the
#  primary tabular data analysis tools in the last few decades. Maybe if the conveniences and gotchas of the Index were
#  better socialized during pandas infancy, we wouldn't see confusion amongst new users- but many people who pick up a dataframe
#  tool simply assume it follows the "rules" of the tool they're familiar with (R, SQL, etc.) and so they fall into the "gotchas".
