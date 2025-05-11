import pandas as pd
import streamlit as st

from pandas import DataFrame

pd_df = DataFrame({
    'Name': ['John', 'John', 'Bob', 'Alice', 'Alice', 'Alice'],
    'Source': ['A', 'B', 'B', 'Z', 'Y', 'X'],
    'Description': ['Text1', 'Longer text', 'Text2', 'Longer text', 'The Longest text', 'Text3'],
    'Value': [1, 4, 2, 5, 3, 6]
})

st.write(pd_df)

st.write('result should look like',DataFrame({
    'Name': ['Alice', 'Bob', 'John'],
    'Source': ['X, Y, Z', 'B', 'A, B'],
    'Description': ['The Longest text', 'Text2', 'Longer text'],
    'Value': [3, 2, 4],
}))

def custom_agg(group):
    longest_desc_df = (
        group.assign(_strlen=group['Description'].str.len())
        .nlargest(1, '_strlen')
    )

    return DataFrame({
        'Source': ', '.join(group['Source'].sort_values()),
        'Description': longest_desc_df['Description'],
        'Value': longest_desc_df['Value']
    })

st.write(pd_df.groupby('Name').apply(custom_agg, include_groups=False).droplevel(1))

def custom_agg(group):
    longest_desc_df = (
        group.assign(_strlen=group['Description'].str.len())
        .nlargest(1, '_strlen')
    )

    return DataFrame({
        'Source': ', '.join(group['Source'].sort_values()),
        'Description': longest_desc_df['Description'],
        'Value': longest_desc_df['Value']
    })

st.write('custom agg',pd_df.groupby('Name').apply(custom_agg, include_groups=False).droplevel(1))

def custom_agg(group):
    longest_desc_loc = (
        group['Description'].str.len() # applied equally regardless of the group
        .idxmax()                      # Series.groupby(…).idxmax(…)
    )

    return DataFrame({ 
        'Source': ', '.join(group['Source'].sort_values()),
        'Description': group.loc[longest_desc_loc, 'Description'],
        'Value': group.loc[longest_desc_loc, 'Value'],
    }, index=[0])

# we’re ultimately aggregating here, use the most appropriate verb `.agg`
# By using this verb we will also avoid accessing multiple columns within the
#   same grouped computation. (`custom_agg` uses "Description", "Source", and "Value").
def naive_pandas(): # put into a function for later comparison
    return pd_df.groupby('Name').apply(custom_agg, include_groups=False).droplevel(1)

st.write('naive pandas', naive_pandas())
# naive_pandas()

def refactored_pandas():
    return (
        pd_df
        .assign( # pre-compute any computation used equally across all groups
            _desc_length=lambda d: d['Description'].str.len(),
        )
        .groupby('Name').agg(
            Source=('Source', lambda g: ', '.join(sorted(g))), # can't avoid a UDF here
            longest_desc_location=('_desc_length', 'idxmax')   # avoided a UDF
        )
        .merge( # fetches the "Description" and "Value" where we observed the longest description
            pd_df.drop(columns=["Name", "Source"]),
            left_on="longest_desc_location",
            right_index=True
        )
        .drop(columns=['longest_desc_location']) # Remove intermediate/temp columns
    )

st.write('refactored pandas',refactored_pandas())

from random import Random
from numpy import unique
from numpy.random import default_rng
from pandas import DataFrame, Series
from string import ascii_uppercase

rnd = Random(0)
rng = default_rng(0)
categories = unique(
    rng.choice([*ascii_uppercase], size=(10_000, length := 4), replace=True)
    .view(f'<U{length}')
)

pd_df = DataFrame({
    'Name'   : categories.repeat(reps := 100),
    'Source' : rng.choice([*ascii_uppercase], size=(len(categories) * reps, 2)).view('<U2').ravel(),
    'Description' : [
        "".join(rnd.choices(ascii_uppercase, k=rnd.randrange(0, 10)))
        for _ in range(len(categories) * reps)
    ],
    'Value' : rng.integers(0, 1_000_000, size=(reps * categories.size)),
}).sample(frac=1, replace=True).reset_index(drop=True)

st.write((f'{len(pd_df) = :,} rows'))
st.write('more cardinality',pd_df.head())

# st.write('unique values',pd_df.apply(lambda g: g.nunique())) # The number of unique values in each column


st.write('another solution',
         
(    
    pd_df
    .sort_values( # sorting is typically expensive, but now we don't need to self join/merge
        "Description", key=lambda g: g.str.len(), ascending=False, kind="mergesort"
    )
    .groupby("Name")
    .agg(
        Source=('Source', lambda g: ', '.join(sorted(g))),
  
        Description=('Description', 'first'),
        Value=('Value', 'first'),
    )
))