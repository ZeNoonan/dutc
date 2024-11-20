import streamlit as st
# import panel as pn
# from panel.pane import Markdown, HTML
from pandas import read_csv, MultiIndex
from ipyvizzu import Chart, Data, Config, Style, DisplayTarget
from numpy import arange, linspace, concatenate
from streamlit.components.v1 import html
from ipyvizzustory import Story, Slide, Step
import ssl

st.set_page_config(layout="wide")

ssl._create_default_https_context = ssl._create_unverified_context


# st.set_page_config(page_title='Title goes here', layout='centered')

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

st.write(medals_count)
st.write(countries_min)

data = Data()
data.add_data_frame(medals_count)

story = Story(data=data)
story.set_size(550, 700)

slide1 = Slide(
    Step(
        # Data,   
        # Data,     
        Config({
                'x': {'set': ['Total Medals']},
                'y': {'set': ['Country']},
                'sort': 'byValue'
            }),
        Style({
            "plot": { 'paddingTop':40, 'paddingLeft': 150,
                "yAxis": { 'title': {'color': '#FFFFFF00' },"label": { 'numberFormat' : 'prefixed','numberScale':'shortScaleSymbolUS'}},
                "xAxis": { 'title': {'color': '#FFFFFF00' }, "label": {"angle": "2.5", 'numberFormat' : 'prefixed','numberScale':'shortScaleSymbolUS'}},
        }
    })
    )
)

# story.add_slide(slide1)

slide2 = Slide(
    Step(
        # Data.filter("record.Country=='United States'||record.Country=='Soviet Union'||record.Country=='Germany'"), THIS WORKS  
        Data.filter(filt),     
        Config({
                'x': {'set': ['Total Medals']},
                'y': {'set': ['Country']},
                'sort': 'byValue'
            }),
        Style({
            "plot": { 'paddingTop':40, 'paddingLeft': 150,
                "yAxis": { 'title': {'color': '#FFFFFF00' },"label": { 'numberFormat' : 'prefixed','numberScale':'shortScaleSymbolUS'}},
                "xAxis": { 'title': {'color': '#FFFFFF00' }, "label": {"angle": "2.5", 'numberFormat' : 'prefixed','numberScale':'shortScaleSymbolUS'}},
        }
    })
    )
)

story.add_slide(slide2)

# Display in Streamlit
html(story._repr_html_(), width=750, height=450)
# the above did work, honestly it did, just need a lot of work to understand the ipyvizzu library and how it works
# not sure i get any real benefit out of it, really depends on the data set

# config = {
#     'channels': {
#         'y': {
#             'set': ['Country'],

#         },
#         'x':{'set': ['Total Medals']}
#     },
#     'sort': 'byValue'
# }

# style=Style(
#     {'plot': {'paddingTop':40, 'paddingLeft': 150}}
# )

# chart = Chart(
#     width="750px",height="900px",
#     display=DisplayTarget.MANUAL
# )

# chart.on('logo-draw','event.preventDefault():')
# chart.animate(
#     data,
#     style,
#     Config(config | {'title': 'United States Leads Summer'})
# )

# filt = '||'.join(
#     f"record.Country=='{c}"
#     for c in countries_min
# )

# chart.animate(
#     Config({
#         'title': 'Countries Winning > 80 Summer Olympic Medals'
#     }),
#     Data.filter(filt),
#     delay=5
# )

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

# import streamlit as st
# from pandas import read_csv, MultiIndex
# from ipyvizzu import Data, Config, Style, DisplayTarget
# from ipyvizzustory import Story, Slide, Step
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title='Title goes here', layout='centered')

# Data preparation
countries = (
    read_csv("C:/Users/Darragh/Documents/Python/dutc/dictionary.csv")
    .set_index('Code')['Country']
)
countries['URS'] = 'Soviet Union'

medals_count = (
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

medals_count['Cumulative Medals'] = medals_count.groupby(['Country'])['Total Medals'].cumsum()

countries_min = (
    medals_count.groupby('Country')
    ['Total Medals'].sum()
    .gt(80)
    .loc[lambda s: s].index
)










# Prepare filtered data
filtered_medals_count = medals_count[medals_count['Country'].isin(countries_min)]

# Create Story
# story = Story(width="750px", height="900px")


# data = Data()
# data.add_data_frame(medals_count)

# # Create Story
# story = Story(data=data)

# # Create the data objects
# data_full = Data()
# data_full.add_data_frame(medals_count)

# data_filtered = Data()
# data_filtered.add_data_frame(filtered_medals_count)

# # Base style that will be used across slides
# base_style = Style({
#     'plot': {
#         'paddingTop': 40,
#         'paddingLeft': 150
#     }
# })

# # Initial slide
# slide1 = Slide(
#     Step(
#         data_full,
#         Config({
#             'channels': {
#                 'y': {'set': ['Country']},
#                 'x': {'set': ['Total Medals']}
#             },
#             'sort': 'byValue',
#             'title': 'United States Leads Summer'
#         }),
#         base_style
#     )
# )

# # Second slide with filtered data
# slide2 = Slide(
#     Step(
#         data_filtered,
#         Config({
#             'title': 'Countries Winning > 80 Summer Olympic Medals',
#             'channels': {
#                 'y': {'set': ['Country']},
#                 'x': {'set': ['Total Medals']}
#             },
#             'sort': 'byValue'
#         }),
#         base_style
#     )
# )

# # Add slides to story
# story.add_slide(slide1)
# story.add_slide(slide2)

# # Create the HTML component
# story_html = story.to_html()

# # Display in Streamlit
# html(story_html, width=800, height=950)































# Create the data object
data = Data()
data.add_data_frame(medals_count)

# Create Story
story = Story(data=data)

# Base style that will be used across slides
base_style = Style({
    'plot': {
        'paddingTop': 40,
        'paddingLeft': 150
    }
})

# Initial slide
slide1 = Slide(
    Step(
        data,
        Config({
            'channels': {
                'y': {'set': ['Country']},
                'x': {'set': ['Total Medals']}
            },
            'sort': 'byValue',
            'title': 'United States Leads Summer'
        }),
        base_style
    )
)

# Filter for countries with more than 80 medals
filter_expression = '||'.join(f"record.Country=='{c}'" for c in countries_min)

# Second slide with filtered data
slide2 = Slide(
    Step(
        Data.filter(filter_expression),
        Config({
            'title': 'Countries Winning > 80 Summer Olympic Medals',
            'channels': {
                'y': {'set': ['Country']},
                'x': {'set': ['Total Medals']}
            },
            'sort': 'byValue'
        })
    )
)

# Add slides to story
story.add_slide(slide1)
story.add_slide(slide2)

# Create the HTML component
# story_html = story.to_html()

# Display in Streamlit
html(story._repr_html_(), width=750, height=450)