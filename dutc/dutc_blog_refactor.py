import pandas as pd
import streamlit as st

# Process A
a_data = [1, 2, 3, 4]                  # ① load
a_transform = [x ** 2 for x in a_data] # ② process
a_result = sum(a_transform)
st.write(f'{a_result = }')                # ③ report


# Process B
b_data = [5, 6, 0, -1]                 # ① load
b_transform = [x ** 2 for x in b_data] # ② process
b_result = sum(b_transform)
st.write(f'{b_result = }')                # ③ report

# Process C
c_data = [-10, 8 , 4, 0]               # ① load
c_transform = [x ** 2 for x in a_data] # ② process
c_result = sum(c_transform)
st.write(f'{c_result = }')                # ③ report

def sum_of_squares(values):
    return sum(x ** 2 for x in values)

# ① load
datasets = {
    'a': [1, 2, 3, 4],
    'b': [5, 6, 0, -1],
    'c': [-10, 8 , 4, 0]
}

# ② process
results = {label: sum_of_squares(values) for label, values in datasets.items()}

# ③ report
for label, result in results.items():
    st.write(f'{label}_result → {result:>3}')

def process_data(label, values):
    if label.casefold() in ('b', 'c'):
        return sum(x ** 3 for x in values)
    return sum(x ** 2 for x in values)

datasets = {
    'a': [1, 2, 3, 4],
    'b': [5, 6, 0, -1],
    'c': [-10, 8 , 4, 0]
}
results = {
    label: process_data(label, values) for label, values in datasets.items()
}
for label, result in results.items():
    st.write(f'{label}_result → {result:>5}')

from collections import defaultdict

def sum_of_cubes(values):
    return sum(x ** 3 for x in values)

def sum_of_squares(values):
    return sum(x ** 2 for x in values)

datasets = datasets = {
    'a': [1, 2, 3, 4],
    'b': [5, 6, 0, -1],
    'c': [-10, 8 , 4, 0]
}

processors = defaultdict(
    lambda: sum_of_squares, # default (in this example, just 'a')
    {
        'b': sum_of_cubes,  # special case 'b'
        'c': sum_of_cubes,  # special case 'c'
    }
)

st.write('claude gave a great write up on the lambda and the benefits of using it')
st.write('the benefit is that by using the lambda it will invoke the function on all datasets except for the ones that are explicity called out like b and c')

st.write('processors', processors)
st.write(sum_of_squares)
# st.write(processors)
results = {
    label: processors[label](values) for label, values in datasets.items()
}
for label, result in results.items():
    st.write(f'{label}_result → {result:>4}')
    
st.write('results dictionary', results)

# https://claude.ai/chat/5b52041f-1146-4bf0-968d-f46ccc2bc352
# Create a sample DataFrame
df = pd.DataFrame({
    'group': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c'],
    'value': [1, 2, 3, 4, 5, 6, 0, -1, -10,8,4,0]
})

st.write("Original DataFrame:")
st.write(df)

# Group by 'group' column
grouped = df.groupby('group')

# Define transformation functions for pandas
def pandas_sum_of_squares(group_df):
    return group_df['value'].pow(2).sum()

def pandas_sum_of_cubes(group_df):
    return group_df['value'].pow(3).sum()

# Create a processor mapping
pandas_processors = defaultdict(lambda: pandas_sum_of_squares)
pandas_processors['b'] = pandas_sum_of_cubes
pandas_processors['c'] = pandas_sum_of_cubes

# Process each group with its corresponding function
pandas_results = {}
for group_name, group_data in grouped:
    pandas_results[group_name] = pandas_processors[group_name](group_data)

st.write("\n pandas results dictionary:", pandas_results)
st.write("\nResults using custom processors:")
for label, result in pandas_results.items():
    st.write(f'{label}_result → {result:>4}')


# Create three separate DataFrames similar to the original datasets
df_a = pd.DataFrame({'value': [1, 2, 3, 4]})
df_b = pd.DataFrame({'value': [5, 6, 0, -1]})
df_c = pd.DataFrame({'value': [-10, 8, 4, 0]})

# Store DataFrames in a dictionary (similar to original datasets)
dataframes = {
    'a': df_a,
    'b': df_b,
    'c': df_c
}

st.write("DataFrame a:")
st.write(df_a)
st.write("\nDataFrame b:")
st.write(df_b)
st.write("\nDataFrame c:")
st.write(df_c)

# Define transformation functions for pandas DataFrames
def pandas_sum_of_squares(df):
    return df['value'].pow(2).sum()

def pandas_sum_of_cubes(df):
    return df['value'].pow(3).sum()

# Create a processor mapping with defaultdict
pandas_processors = defaultdict(lambda: pandas_sum_of_squares)
pandas_processors['b'] = pandas_sum_of_cubes
pandas_processors['c'] = pandas_sum_of_cubes

# Process each DataFrame with its corresponding function
pandas_results = {
    label: pandas_processors[label](df) for label, df in dataframes.items()
}

st.write("\nResults using DataFrame processors:")
st.write("\n pandas results dictionary:", pandas_results)
# for label, result in pandas_results.items():
#     st.write(f'{label}_result → {result:>4}')
