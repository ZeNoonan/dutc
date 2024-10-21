import pandas as pd
import numpy as np
import streamlit as st
from itertools import islice, tee, groupby
from numpy import arange

st.set_page_config(layout="wide")

# https://www.dontusethiscode.com/blog/2024-07-24_pandas-concat.html

data = arange(0, 18).reshape(3, 3, 2)

# take note where these indexes and columns overlap and where they do not
dfs = {
    'df1': pd.DataFrame(data[0], index=['a', 'b', 'c'          ], columns=['x', 'y'     ]),
    'df2': pd.DataFrame(data[1], index=[     'b', 'c', 'd'     ], columns=['x',      'z']),
    'df3': pd.DataFrame(data[2], index=[     'b',      'd', 'e'], columns=['x',      'z']),
}

for k, df in dfs.items():
    st.write(k, df)

with st.echo():
    # align vertically, preserving fully overlapping columns indices
    st.write(pd.concat(dfs, axis='rows', join='outer'))

with st.echo():
    # align vertically, preserving ONLY shared columns indices
    st.write(pd.concat(dfs, axis='rows', join='inner'))

with st.echo():
    # align horizontally, preserving all row indices
    st.write(pd.concat(dfs, axis='columns', join='outer'))

with st.echo():
    # align horizontally, preserving ONLY shared row indices
    st.write(pd.concat(dfs, axis='columns', join='inner'))

from functools import reduce
import pandas as pd

with st.echo():
    parts = [
        #                    highlighting the alignments
        #                            a    b    c    d    e
        pd.Series([0, 1, 2], index=['a', 'b', 'c'          ], dtype='Int64'),
        pd.Series([3, 4, 5], index=[     'b', 'c', 'd'     ], dtype='Int64'),
        pd.Series([6, 7, 8], index=[     'b',      'd', 'e'], dtype='Int64'),
    ]
    st.write('parts',parts)
    indices = [s.index for s in parts]
    st.write('indices',indices)

with st.echo():
    st.write(
        (outer := reduce(lambda l, r: l.union(r),        indices)), # outer
        (inner := reduce(lambda l, r: l.intersection(r), indices)), # inner
    )
    st.write('walrus operator very useful')
    st.write("Without the walrus operator, you'd have to write this code in a more verbose way:")
    # outer = reduce(lambda l, r: l.union(r), indices)
    # inner = reduce(lambda l, r: l.intersection(r), indices)
    # st.write(outer, inner)

with st.echo():
    # outer aligned
    st.write(pd.DataFrame({i: p.reindex(outer) for i, p in enumerate(parts)}))
    # st.write('parts:', parts)
    st.write('outer', outer)
    st.write('very interesting on the .union, this is a pandas function applicable to index - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.union.html')
    for i, p in enumerate(parts):
        st.write('i:', i)
        st.write('parts pandas series with a-e index and numbers:', parts)
        st.write('p reindex outer',p.reindex(outer))
        st.write('dictonary',{i: p.reindex(outer) for i, p in enumerate(parts)})
        print({i: p.reindex(outer) for i, p in enumerate(parts)})


with st.echo():
    # https://docs.python.org/3/library/functools.html#functools.reduce
    def reduce_code(function, iterable, initializer=None):
        it = iter(iterable)
        if initializer is None:
            value = next(it) 
            
        else:
            value = initializer
        for element in it:
            # st.write('value',value)
            # st.write('element',element) # THIS WAS INTERESTING WASNT EXPECTING elment to have number 4, the key is that next(it) produces first number and its consumed ie a generator!!
            value = function(value, element)
            # st.write('value after',value)
        return value

with st.echo():
    st.write(reduce_code(lambda x, y: x+y, [5, 4, 3, 2, 1])) #calculates ((((1+2)+3)+4)+5)

with st.echo():    
    def sum_test_darragh(a,b):
        return a + b
    st.write(" so basically reduce from functools takes a function which can only have 2 inputs and it processes that function over the list given")
    st.write(reduce_code(sum_test_darragh, [5, 4, 3, 2, 1])) #calculates ((((1+2)+3)+4)+5)

with st.echo():
    st.write(reduce_code(lambda l, r: l.union(r),        indices)) # oh cool it works wasnt expecting that :-)

with st.echo():
    st.write('I was confused by the fact that the iterable is a list of lists, how did the reduce_code function process that - time to test it')
    st.write('')
    # for value in iter(indices):
    #     st.write('value',value)
    #     st.write('next.....',next(iter(indices)))
    # x=iter(indices)
    # st.write(next(x))
    # st.write(next(x))
    # st.write(next(x))

    def reduce_code_test(function, iterable, initializer=None):
        it = iter(iterable)
        if initializer is None:
            value = next(it) 
            st.write('value next(it):',value)
            
        else:
            value = initializer
        for element in it:
            st.write('element:',element)
            # st.write('value',value)
            # st.write('element',element) # THIS WAS INTERESTING WASNT EXPECTING elment to have number 4, the key is that next(it) produces first number and its consumed ie a generator!!
            value = function(value, element)
            st.write('value after function',value)
        return value
    
    
    # st.write('indices',indices)
    # st.write(reduce_code_test(lambda l, r: l.union(r),        indices)) # had to run this to understand whats going on

    st.write(reduce_code_test(sum_test_darragh, [[5, 4], [3, 2], [1,1]]))

with st.expander('Constructing a DataFrame using Python'):
    st.write('was just wondering if anything to be gained by understanding the deeper structure of pandas dataframe and how it works')
    with st.echo():
        df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c'],
        'C': [4.0, 5.0, 6.0]
    }, index=['x', 'y', 'z'])

    dict_of_series = df.to_dict('series')

    # Examine the output
    for key, output in dict_of_series.items():
        st.write(f"Key: {key}")
        st.write(f"Type of output: {type(output)}")
        st.write(f"Content of output:\n{output}\n")    

    for key, output in df.to_dict('series').items():
        st.write(key, output.tolist())

    for key, output in df.to_dict('series').items():
        st.write(key, output.to_dict())

    with st.echo():
        # Pseudo-code for generating a Pandas DataFrame using pure Python
        # Define a list of dictionaries, where each dictionary represents a row in the DataFrame
        data = [
            {'name': 'Alice', 'age': 25, 'city': 'New York'},
            {'name': 'Bob', 'age': 32, 'city': 'London'},
            {'name': 'Charlie', 'age': 41, 'city': 'Paris'},
            {'name': 'David', 'age': 28, 'city': 'Tokyo'}
        ]

        # Create a set of unique column names from the data
        columns = set()
        for row in data:
            columns.update(row.keys())

        # Create a dictionary to store the column data
        column_data = {col: [] for col in columns}

        # Populate the column data
        for row in data:
            for col in columns:
                column_data[col].append(row.get(col, None))

        # Create a list of row labels (indices)
        indices = list(range(len(data)))

        # Construct the DataFrame
        dataframe = {
            'columns': list(columns),
            'data': column_data,
            'index': indices
        }