
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

# https://stackoverflow.com/questions/78906297/how-to-merge-lists-of-multiple-data-frames-and-decreasing-the-second-lists-of-da/78906595#78906595
st.write('overall interesting, i think the concat inner is dangerous as joins on union of ID, but what happens if the ID is not in the other df? but it is')
st.write('in the original df, just means i consider the original df to be the single source of truth, and the other df to be the one that is being merged' \
' into the original df')
st.write('this question is also lent itself well to the dynamic use of the enumerate, might not always be like that')
st.write('just on the concat, just wondering would it be better to use the merge and merge on how=left that way i am always getting the original df as the source of truth' \
' and the other df as the one that is being merged into the original df')

df1 = pd.DataFrame({"Id":[101,102,103,104],"Name":["Harish","Hari","Harry",""],"Age":[30,31,32,33],"Mobile":[1,2,3,3]},index=[0,1,2,3])
df2 = pd.DataFrame({"Id":[101,102,103,104],"Name":["Harish","Hari","Harry",""],"Age":[30,31,32,33],"Mobile":[1,2,3,3]},index=[0,1,2,3])
df3 = pd.DataFrame({"Id":[101,102,103,104],"Name":["Harish","Hari","Harry",""],"Age":[30,31,32,33],"Mobile":[1,2,3,3]},index=[0,1,2,3])

df4 = pd.DataFrame({"Id":[101,102,103,104],"ename":["Harish","Hari","Harry",""],"pg":[30,31,32,33],"M+":[1,2,3,3]},index=[0,1,2,3])
df5 = pd.DataFrame({"Id":[101,102,103,104],"en":["Harish","Hari","Harry",""],"pg+":[30,31,32,33],"Tm":[1,2,3,3]},index=[0,1,2,3])
df6 = pd.DataFrame({"Id":[101,102,103,104],"years":["2016","2018","2019",""],"lev":[30,31,32,33],"d+":[1,2,3,3]},index=[0,1,2,3])

data_frames1 = [df1,df2,df3]
data_frames2 = [df4,df5,df6]
st.write('data_frames1', data_frames1)
st.write('data_frames2', data_frames2)

# set index of all frames to the column to join on
data_frames1 = [df.set_index('Id') for df in data_frames1]
data_frames2 = [df.set_index('Id') for df in data_frames2]

# concatenate each df from data_frames1 to the subset of data_frames2
# could also be done as a list comprehension.
results = []
for i, df in enumerate(data_frames1):
    results.append(
        pd.concat([df, *data_frames2[i:]], axis=1, join='inner')
    )
    st.write('in the loop',results)

# st.write(*results, sep='\n' * 2)
st.write(*results)
# print(*results, sep='\n' * 2)
#        Name  Age  Mobile   ename  pg  M+      en  pg+  Tm years  lev  d+
# Id
# 101  Harish   30       1  Harish  30   1  Harish   30   1  2016   30   1
# 102    Hari   31       2    Hari  31   2    Hari   31   2  2018   31   2
# 103   Harry   32       3   Harry  32   3   Harry   32   3  2019   32   3
# 104           33       3          33   3           33   3         33   3

#        Name  Age  Mobile      en  pg+  Tm years  lev  d+
# Id
# 101  Harish   30       1  Harish   30   1  2016   30   1
# 102    Hari   31       2    Hari   31   2  2018   31   2
# 103   Harry   32       3   Harry   32   3  2019   32   3
# 104           33       3           33   3         33   3

#        Name  Age  Mobile years  lev  d+
# Id
# 101  Harish   30       1  2016   30   1
# 102    Hari   31       2  2018   31   2
# 103   Harry   32       3  2019   32   3
# 104           33       3         33   3

st.write('below was from another answer to test output')
from functools import reduce

assert len(data_frames1) == len(data_frames2)

out = [
    reduce(
        lambda a, b: a.merge(b, how='left', on='Id'),
        data_frames2[i:],
        data_frames1[i],
    )
    for i in range(len(data_frames1))
]

m1 = df1.merge(df4,how="left",on="Id")
m2 = m1.merge(df5,how="left",on="Id")
m3 = m2.merge(df6,how="left",on="Id") #-> This is my one final df
m4 = df2.merge(df5,how="left",on="Id")
m5 = m4.merge(df6,how="left",on="Id") #-> This is my second final df
m6 = df3.merge(df6,how="left",on="Id") #-> This my third final df

# checking the output is as expected
all(a.equals(b) for a,b in zip(out, [m3, m5, m6]))
# True

