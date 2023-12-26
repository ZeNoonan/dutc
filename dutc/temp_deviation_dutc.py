import pandas as pd
from numpy import linspace
import streamlit as st
from requests import get
from datetime import date as dt_date
from itertools import pairwise
from polars import Config, DataFrame, col,Datetime
from polars.selectors import starts_with

from matplotlib import colormaps
from matplotlib.pyplot import subplots, rc,close
from matplotlib.colors import to_hex,LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MultipleLocator, NullFormatter
from matplotlib.dates import MonthLocator, DateFormatter
from flexitext import flexitext


st.set_page_config(layout="wide")



location_resp = get(
    'https://nominatim.openstreetmap.org/search',
    params={
        'q': 'Sacramento, CA',
        'format': 'json',
        'limit': 1,
        'addressdetails': 1,
    }
)

location_resp.raise_for_status()
location_data = location_resp.json()[0]

location_data


weather_resp = get(
    "https://archive-api.open-meteo.com/v1/archive", 
    timeout=30,
    params={
        'latitude': location_data['lat'],
        'longitude': location_data['lon'],
        'start_date': (start_date := dt_date(1961, 1, 1)).strftime('%Y-%m-%d'),
        'end_date': (end_date := dt_date(2022, 12, 31)).strftime('%Y-%m-%d'),
        'daily': (metric := 'temperature_2m_mean'),
        'timezone': 'auto',
        'temperature_unit': 'fahrenheit',
    }
)

weather_resp.raise_for_status()
weather_data = weather_resp.json()

weather_df_raw = (
    DataFrame({
        'date': weather_data['daily']['time'],
        'temp': weather_data['daily'][metric],
    })
    .with_columns(
        date=col('date').str.to_date('%Y-%m-%d'),
    )
    .with_columns(
        year=col('date').dt.year(), 
        month=col('date').dt.month(), 
        day=col('date').dt.day()
    )
)

with Config(tbl_rows=6):
    # display(weather_df_raw)
    # st.pyplot(weather_df_raw)
    st.write(weather_df_raw)

ref_years = (start_date.year, 1990)
show_year = 2022

weather_df_ref = (
    weather_df_raw
    .filter(col('year').is_between(*ref_years))
    .group_by(['month', 'day'])
    .agg(
        temp_ref_lb=col('temp').quantile((lb_value := .05)),
        temp_ref_ub=col('temp').quantile((ub_value := .95)),
        temp_ref_avg=col('temp').mean(),
    )
    .sort(['month', 'day'])
)

weather_df_ref.head()



weather_df_plotting = (
    weather_df_ref
    .join(
        weather_df_raw.filter(col('date').dt.year() == show_year), 
        on=['month', 'day'], how='left'
    )
    .drop_nulls()
    .with_columns(col('date').cast(Datetime))
    .with_columns(
        starts_with('temp')
            .rolling_mean('3d', by='date', closed='right', min_periods=1)
    )
    .with_columns(
        distance=(
            (col('temp') - col('temp_ref_avg'))
            .pipe(lambda c: c / c.abs().max())
        ),
    )
)

weather_df_plotting.head()



pdf = weather_df_plotting

rc('font', size=16)
rc('axes.spines', left=False, top=False, bottom=False, right=False)

fig, ax = subplots(figsize=(18, 6))

lb_line, = ax.plot('date', 'temp_ref_lb', color='k', data=pdf, ls='--', lw=.7, zorder=6)
ub_line, = ax.plot('date', 'temp_ref_ub', color='k', data=pdf, ls='--', lw=.7, zorder=6)
context_pc = ax.fill_between(
    'date', 'temp_ref_lb', 'temp_ref_ub', data=pdf, color='gainsboro', ec='none', zorder=5
)

ax.plot('date', 'temp_ref_avg', data=pdf, color='k', lw=1, zorder=6)
# display(fig)
st.pyplot(fig)



# Use the `RdBu` colormap, removing the very dark shades at either end
colors = colormaps['RdBu_r'](linspace(0, 1, 256))[30:-30]
cmap = LinearSegmentedColormap.from_list('cut_RdBU_r', colors)

raw_pc = ax.fill_between( # the raw data are not visible (alpha=0)
    'date', 'temp', 'temp_ref_avg', data=pdf, ec='none', alpha=0
)

arr = raw_pc.get_paths()[0].vertices
(x0, y0), (x1, y1) = arr.min(axis=0), arr.max(axis=0)

gradient = ax.imshow(
    pdf.select(col('distance')).to_numpy().reshape(1, -1),
    extent=[x0, x1, y0, y1],
    aspect='auto',
    cmap=cmap,
    norm=TwoSlopeNorm(0), # should color intensities be symmetrical around 0?
    interpolation='bicubic',
    zorder=5,
    
    # redundant encoding of color on alpha. Focus on extreme values.
    alpha=pdf.select(col('distance').abs().sqrt()).to_numpy().reshape(1, -1),
)

st.pyplot(fig)

# use the raw `fill_between` object (PolyCollection)
#  to mask the gradient generated from imshow
gradient.set_clip_path(raw_pc.get_paths()[0], transform=ax.transData)
ax.use_sticky_edges = False
ax.margins(x=0, y=.01)

st.pyplot(fig)


ax.yaxis.grid(True)
ax.yaxis.set_tick_params(left=False)
ax.yaxis.set_major_formatter(lambda x, pos: f'{x:g}°F')

ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_major_formatter(NullFormatter())
ax.xaxis.set_minor_locator(MonthLocator(bymonthday=16))
ax.xaxis.set_minor_formatter(DateFormatter('%B'))
ax.xaxis.set_tick_params(which='both', bottom=False, labelsize='small')

for i, (left, right) in enumerate(pairwise(ax.get_xticks())):
    if i % 2 == 0:
        ax.axvspan(left, right, 0, 1, color='gainsboro', alpha=.2)

st.pyplot(fig)

def text_in_axes(*text, ax):
    """Force all Text objects to appear inside of the data limits of the Axes.
    """
    for t in text:
        bbox = t.get_window_extent()
        bbox_trans = bbox.transformed(ax.transData.inverted())
        ax.update_datalim(bbox_trans.corners())

ref_span_annot = ax.annotate(
    f'{round((ub_value-lb_value)*100)}% of reference period data fall within the gray area',
    xy=(
        ax.xaxis.convert_units(x := dt_date(2022, 7, 1)), 
        pdf.filter(col('date') == x)
        .select((col('temp_ref_ub') - col('temp')) / 2 + col('temp'))
        .to_numpy().squeeze()
    ),
    xycoords=ax.transData,
    xytext=(
        .55, pdf.filter(col('date') >= x).select(['temp', 'temp_ref_ub']).to_numpy().max()
    ),
    textcoords=ax.get_yaxis_transform(),
    va='bottom',
    ha='left',
    arrowprops=dict(lw=1, arrowstyle='-', relpos=(0, 0)),
    zorder=10,
    fontsize='small',
)

ref_avg_annot = ax.annotate(
    f'Mean temperature {ref_years[0]}-{ref_years[1]}',
    xy=(
        (x := dt_date(2022, 2, 1)), 
        pdf.filter(col('date') == x).select(['temp_ref_avg']).to_numpy().squeeze()
    ),
    xytext=(.15, .09), textcoords=ax.transAxes,
    va='top',
    ha='left',
    arrowprops=dict(lw=1, arrowstyle='-', relpos=(0, .7), shrinkB=0),
    zorder=10,
    fontsize='small',
)

text_in_axes(ref_span_annot, ref_avg_annot, ax=ax)

for line, label in {lb_line: 'P05', ub_line: 'P95'}.items():
    ax.annotate(
        text=label,
        xy=line.get_xydata()[-1],
        xytext=(5, 0), textcoords='offset points',
        va='center',
        size='small',
    )

ax.autoscale_view()
st.pyplot(fig)



ad = location_data['address']
blue, red = cmap(0), cmap(cmap.N)

flexitext(
    s=(
        f'<size:large,weight:semibold>{show_year} <color:{to_hex(red)}>Hot</> '
        f' and <color:{to_hex(blue)}>Cold</> Temperature Deviations from'
         ' Historical Average\n</>'
        f'<size:medium>{ad["city"]} {ad["state"]}, {ad["country"]}</>'
    ),
    x=0, y=1.01,
    va='bottom',
    ax=ax
)

st.pyplot(fig)

# from matplotlib.pyplot import close
close('all')


