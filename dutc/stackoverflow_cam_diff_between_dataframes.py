import pandas as pd
import streamlit as st

# https://stackoverflow.com/questions/78570861/smarter-way-to-create-diff-between-two-pandas-dataframes

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


dir_old = pd.DataFrame([
    {"Filepath": "Karmas", "Hash": 2020},
    {"Filepath": "Friends in Oz", "Hash": 2021},
    {"Filepath": "Seuss", "Hash": 2022},
])

dir_new = pd.DataFrame([
    # {"Filepath": "dir1/file1", "Hash": "hash1"}, # deleted file
    {"Filepath": "Friends in Oz", "Hash": 2021},
    {"Filepath": "Seuss", "Hash": 2024},  # changed file
    {"Filepath": "Robo", "Hash": 2023},  # new file
])

df_merged = pd.merge(
    dir_new, dir_old, on='Filepath', how='outer', indicator=True
)

st.write('dir_old', dir_old, 'dir_new', dir_new)
st.write('df merged', df_merged)
st.write('change up merge outer',pd.merge(
    dir_new, dir_old, on=['Filepath','Hash'], how='outer', indicator=True
))
st.write('change up merge left',pd.merge(
    dir_new, dir_old, on=['Filepath','Hash'], how='left', indicator=True
))


st.write(
    df_merged
    .assign(
        Hash=lambda d: d['Hash_x'].fillna(d['Hash_y']),
        Status=lambda d:  # NA is our fallthrough value (if cases are not exhaustive)
            pd.Series(pd.NA, index=d.index, dtype='string')
            .case_when([
                (d['_merge'] == 'right_only',                          'deleted'  ),
                (d['_merge'] == 'left_only',                           'created'  ),
                (d['_merge'].eq('both') & d['Hash_x'].ne(d['Hash_y']), 'changed'  ),
                (d['_merge'].eq('both') & d['Hash_x'].eq(d['Hash_y']), 'unchanged'),
            ]),
    )
    .drop(columns=['Hash_x', 'Hash_y', '_merge'])
)
#      Filepath   Hash     Status
# 0  dir1/file1  hash1    deleted
# 1  dir1/file2  hash2  unchanged
# 2  dir1/file4  hash4    created
# 3  dir2/file3  hash5    changed