import pandas as pd
import numpy as np
import streamlit as st
from itertools import islice, tee, groupby
st.set_page_config(layout="wide")

# https://www.dontusethiscode.com/blog/2024-07-31_pandas-groupby-axis.html

grades = pd.DataFrame({
    'student'     : [*'ABC'],
    'math_1'      : [99, 72, 12],
    'math_2'      : [pd.NA, pd.NA, 63],
    'literature_1': [45, 78, 96],
    'literature_2': [80, 83, pd.NA],
    'PE'          : [56, 55, 57]
})

st.write(grades)

groupings = {
    'math'      : ['math_1', 'math_2'],
    'literature': ['literature_1', 'literature_2'],
    'PE'        : ['PE'],
}

groupings_map = {v_: k for k, v in groupings.items() for v_ in v}
st.write('groupings map used in grouping?!!',groupings_map)

# now we can perform our groupby operation!
with st.echo():
    st.write('result old way',grades.set_index('student').groupby(groupings_map, axis=1).max())
    st.write('1. Transform',grades.set_index('student').T)
    st.write('2. Transform and groupby on groupings map final result',grades.set_index('student').T.groupby(groupings_map).max())
    st.write('3. Transformed after groupby',grades.set_index('student').T.groupby(groupings_map).max().T)

# st.write(grades.set_index('student').stack())
# st.write('index',grades.set_index('student').stack().index)
# st.write('index names',grades.set_index('student').stack().index.names)
# st.write(grades.set_index('student').stack().rename_axis([*grades.index.names,"subject"]).rename("grade"))
# st.write(grades.set_index('student').stack().rename_axis([*grades.index.names,"subject"]).rename("grade").index.names)
st.write('this is my version of answer using unstack dont like the way its quite long and it uses reset index',
         grades.set_index('student').stack().rename_axis(index={'student': 'student', None: 'subject'}).rename("grade")
         .to_frame().assign(subject_map=lambda x: x.index.get_level_values('subject').map(groupings_map)).reset_index()
         .groupby(['student','subject_map'])['grade'].max().unstack()
         )
st.write('how about this index names',grades.set_index('student').stack().rename_axis(index={'student': 'student', None: 'subject'}).rename("grade").index.names)
# st.write('how about this groupby',grades.set_index('student').stack().rename_axis(index={'student': 'student', None: 'subject'}).rename("grade").set_index('subject').groupby(groupings_map).max()) # NOT WORKING
# st.write('how about this groupby',grades.set_index('student').stack().rename_axis(index={'student': 'student', None: 'subject'}).rename("grade")\
#          .groupby(by=['student','subject']).max())
# st.write('how about this groupby x2',grades.set_index('student').stack().rename_axis(index={'student': 'student', None: 'subject'}).rename("grade")
#          .to_frame().assign(subject=lambda x: x.index.get_level_values('subject').map(groupings_map)).columns)
st.write('still getting confused with .stack() just not doing what i expect maybe go back to pure python code to understand the nuances')

# st.write('reset_index',grades.set_index('student').stack().rename_axis([*grades.index.names,"subject"]).rename("grade").reset_index())

with st.expander('actual answer'):
    st.write('original dataframe - grades:  ',grades)
    groupings = {
    'math'      : ['math_1', 'math_2'],
    'literature': ['literature_1', 'literature_2'],
    'PE'        : ['PE'],
    }
    st.write('This is the groupings: ', groupings)
    results = {}
    for result_name, columns in groupings.items():
        st.write('result name: ', result_name)
        st.write( 'columns: ', columns)
        st.write('grades[columns]: ', grades[columns])
        results[result_name] = grades[columns].max(axis=1) # max along the columns
    st.write( pd.concat(results, axis=1).set_axis(grades['student']) )

    # column_groupings = groupby(
    # sorted([col for col in grades.columns if col not in ['student']]),
    # key=lambda name: name.split('_')[0]
    # )

    # for prefix, columns in column_groupings:
    #     print(f'{prefix:<10} {[*columns]}')

    # column_groupings = groupby(
    # sorted([col for col in grades.columns if col not in ['student']]),
    # key=lambda name: name.split('_')[0]
    # )

    # for result_name, columns in column_groupings:
    #     results[result_name] = grades[[*columns]].max(axis=1)
    # pd.concat(results, axis=1).set_axis(grades['student'])