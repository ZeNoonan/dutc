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
print(sum_of_squares)
# print(processors)
results = {
    label: processors[label](values) for label, values in datasets.items()
}
for label, result in results.items():
    st.write(f'{label}_result → {result:>4}')