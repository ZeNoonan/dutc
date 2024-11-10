import streamlit as st
import panel as pn
from panel.pane import Markdown, HTML
from pandas import read_csv, MultiIndex
from ipyvizzu import Chart, Data, Config, Style, DisplayTarget
from numpy import arange, linspace, concatenate
from streamlit.components.v1 import html
from ipyvizzustory import Story, Slide, Step
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# st.set_page_config(layout="wide")
st.set_page_config(page_title='Title goes here', layout='centered')

countries = (
    read_csv("C:/Users/Darragh/Documents/Python/dutc/dictionary.csv")
    .set_index('Code')['Country']
)
countries['URS'] = 'Soviet Union'

medals_count= (
    read_csv("C:/Users/Darragh/Documents/Python/dutc/summer.csv")
    .drop_duplicates(['Year','Country','Event','Medal'])
    .groupby(['Year','Country']).size()
    .rename('Total Medals')
)

medals_count = (
    medals_count.reindex(
        MultiIndex.from_product(
            medals_count.index.levels),
            fill_value=0
        )
        .reset_index()
        .astype({'Year':str})
        .assign(Country=lambda d: d['Country'].map(countries)
    )
)

# used claude to complete code
medals_count['Cumulative Medals'] = medals_count.groupby(['Country'])['Total Medals'].cumsum()

countries_min = (
    medals_count.groupby('Country')
    ['Total Medals'].sum()
    .gt(80)
    .loc[lambda s: s].index
)

data = Data()
data.add_data_frame(medals_count)

config = {
    'channels': {
        'y': {
            'set': ['Country'],

        },
        'x':{'set': ['Total Medals']}
    },
    'sort': 'byValue'
}

style=Style(
    {'plot': {'paddingTop':40, 'paddingLeft': 150}}
)

chart = Chart(
    width="750px",height="900px",
    display=DisplayTarget.MANUAL
)

chart.on('logo-draw','event.preventDefault():')
chart.animate(
    data,
    style,
    Config(config | {'title': 'United States Leads Summer'})
)

filt = '||'.join(
    f"record.Country=='{c}"
    for c in countries_min
)

chart.animate(
    Config({
        'title': 'Countries Winning > 80 Summer Olympic Medals'
    }),
    Data.filter(filt),
    delay=5
)

# for year, group in medals_count.groupby('Year'):
#     title = 'Summer Olympics Medals 1896'
#     if year != '1896':
#         title += f' - {year}'
#     chart.animate(
#         Data.filter(
#             f'record.Year == {year} && ({filt})'
#         ),
#         Config(
#             config |
#             {'title': title, 'x': 'Cumulative Medals'}
#         ),
#         duration=1,
#         x={"easing": "linear","delay":0},
#         y={"delay":0},
#         show={"delay":0},
#         hide={"delay":0},
#         title={"duration":0, "delay":0},

#     )

# pn.Column(
#     pn.Column(HTML(chart))
# ).servable()

# st.write(countries)
# st.write(medals_count)