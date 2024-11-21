import pandas as pd
import streamlit as st
from pandas import DataFrame,MultiIndex,period_range,IndexSlice, Series,merge,concat, get_dummies, NA, Timestamp, to_timedelta, concat as pd_concat, CategoricalIndex,date_range

st.set_page_config(layout="wide")

def highlight_index(index, mapping={}):
    return index.map({
        k: f'background-color: {v}; color: black;' 
        for k, v in mapping.items()
    }).fillna('')

def highlight_columns(df, mapping):
    def _highlight_col(s):
        color = mapping.get(s.name, '')
        return [f"background-color: {color}; color: black"] * len(df)
    
    return (
        df.style.apply(_highlight_col, subset=[*mapping.keys()])
        .apply_index(highlight_index, axis=1, mapping=mapping)
    )

def highlight_rows(df, mapping, subset=[]):
    result = None
    mapping = {
        k: f'background-color: {v}; color: black' 
        for k, v in mapping.items()
    }
    def _highlight_col(s):
        nonlocal result
        if result is not None:
            return result
        result = s.map(mapping).fillna('')
        return result
    
    return (
        df.style.apply(_highlight_col, subset=subset)
    )


from pandas import DataFrame

df = DataFrame({
    'product': [*'ABCD'],
    'jan'    : [NA, 11, 12, 13],
    'feb'    : [14, 15, 16, 17],
    'mar'    : [18, 19, NA, NA],
}).convert_dtypes()
color_mapping = {'jan': '#66D9EF', 'feb': '#FF6F61', 'mar': '#FFF7E5'}

st.write(
    df.pipe(highlight_columns, mapping=color_mapping),
    
    df.melt(id_vars='product', var_name='month', value_name='sales')
    .pipe(highlight_rows, mapping=color_mapping, subset=['month', 'sales'])
)

from pandas import DataFrame, NA
from pandas import CategoricalDtype
from calendar import month_abbr

MonthDtype = CategoricalDtype([m.lower() for m in month_abbr[1:]], ordered=True)

df = (
    DataFrame({
        'product': [*'AABBBCCDD'],
        'month'  : [*'fmjfmjfjf'], 
        'sales'  : [*range(11, 11+9)],
    })
    .assign( # replace 'j' → 'jan', etc.
        month=lambda d: d['month'].map({'j': 'jan', 'f': 'feb', 'm': 'mar'})
    )
    .astype({'month': MonthDtype, 'sales': 'Int64'})
    .sort_values(['month', 'product'])
    .reset_index(drop=True)
)

st.write(
    df.pipe(highlight_rows, mapping=color_mapping, subset=['month', 'sales']),
    
    df.pivot(index='product', columns='month', values='sales')
    .rename_axis(columns=None).reset_index()
    .pipe(highlight_columns, mapping=color_mapping)
)