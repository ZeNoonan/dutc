


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

df_merged = pd.merge(
    dir_new, dir_old, on='Filepath', how='outer', indicator=True
)

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