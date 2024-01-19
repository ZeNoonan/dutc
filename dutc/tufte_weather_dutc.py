from pandas import read_csv,to_datetime, to_timedelta,read_parquet
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

dtype={
    'station': 'category', 
    'date': 'string', # later converted to datetime,
    'measurement': 'category',
    'value': 'int',
    'm_flag': 'category',
    'q_flag': 'category',
    's_flag': 'category',
    'obs_time': 'string',
}

# nyc_weather = (
#     read_csv(
#         "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00014732.csv.gz",
#         header=None, names=dtype.keys(), usecols=[*range(len(dtype))],
#         parse_dates=['date'], infer_datetime_format=True,
#         dtype=dtype
#     )
# )

# nyc_weather.to_parquet('data/NYC_weather.parquet')

st.write('lets have a look at the raw data')
# st.write(pd.read_csv("https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/USW00014732.csv.gz",header=None).head())

# st.write(read_parquet('data/NYC_weather.parquet',columns=['date', 'measurement', 'value', 'm_flag', 'q_flag']).loc[lambda df: 
#          ~df['q_flag'].isin(['I', 'W', 'X'])])

st.write('original data',read_parquet('data/NYC_weather.parquet',columns=['date', 'measurement', 'value', 'm_flag', 'q_flag']))
st.write('original data sum',read_parquet('data/NYC_weather.parquet',columns=['date', 'measurement', 'value', 'm_flag', 'q_flag'])['value'].sum())
# test_df=read_parquet('data/NYC_weather.parquet',columns=['date', 'measurement', 'value', 'm_flag', 'q_flag'])
test_pivot=read_parquet(
        'data/NYC_weather.parquet',
        columns=['date', 'measurement', 'value', 'm_flag', 'q_flag']).loc[lambda df: 
         ~df['q_flag'].isin(['I', 'W', 'X'])
         & df['m_flag'].isna()
         & df['measurement'].isin(['PRCP', 'TMAX', 'TMIN', 'SNOW'])].pivot(index='date', columns='measurement', values='value')
st.write('sum of pivot', test_pivot.loc[:,['PRCP','TMAX', 'TMIN', 'SNOW']].sum().sum())
st.write('should there be a diff?')


st.write('pivot',read_parquet(
        'data/NYC_weather.parquet',
        columns=['date', 'measurement', 'value', 'm_flag', 'q_flag'],
    )
    .loc[lambda df: 
         ~df['q_flag'].isin(['I', 'W', 'X'])
         & df['m_flag'].isna()
         & df['measurement'].isin(['PRCP', 'TMAX', 'TMIN', 'SNOW'])
    ]
    .pivot(index='date', columns='measurement', values='value'))



nyc_historical = (
    read_parquet(
        'data/NYC_weather.parquet',
        columns=['date', 'measurement', 'value', 'm_flag', 'q_flag'],
    )
    .loc[lambda df: 
         ~df['q_flag'].isin(['I', 'W', 'X'])
         & df['m_flag'].isna()
         & df['measurement'].isin(['PRCP', 'TMAX', 'TMIN', 'SNOW'])
    ]
    .pivot(index='date', columns='measurement', values='value')
    .eval('''
        TMAX = 9/5 * (TMAX/10) + 32
        TMIN = 9/5 * (TMIN/10) + 32
        PRCP = PRCP / 10 / 25.4
        SNOW = SNOW / 25.4
    ''')
    .rename(columns={               # units post-conversion
        'TMAX': 'temperature_max',  # farenheit
        'TMIN': 'temperature_min',  # farenheit
        'PRCP': 'precipitation',    # inches
        'SNOW': 'snowfall'          # inches
    })
    .rename_axis(columns=None)
    .sort_index()
)

nyc_2003 = nyc_historical.loc['2003'].copy()
st.write(nyc_2003)

historical_range = (
    nyc_historical.groupby(nyc_historical.index.dayofyear)
    .agg(
        historical_min=('temperature_min', 'min'), 
        historical_max=('temperature_max', 'max'),
        normal_min=('temperature_min', 'mean'), 
        normal_max=('temperature_max', 'mean'),
    )
)

historical_range.head()

historical_align_index = (
    to_datetime('2002-12-31') + to_timedelta(historical_range.index, unit='D')
)

plot_data = (
    nyc_2003.join(historical_range.set_index(historical_align_index))
    .assign(
        monthly_cumul_precip=lambda d: 
            d.fillna({'precipitation': 0})
            .resample('M')['precipitation']
            .cumsum()
   )
)

plot_data.head()

from matplotlib.pyplot import close
close('all')
from matplotlib.pyplot import figure, GridSpec, rc, rcdefaults, close, setp
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.ticker import FixedLocator, MultipleLocator
from matplotlib.transforms import blended_transform_factory, offset_copy

# Some colors I grabbed from the original plot. There aren't many as color can
#  distract from the overall message being communicated.
palette = {
    'background': '#e5e1d8', 
    'daily_range': '#5f3946',
    'record_range': '#c8c0aa',
    'normal_range': '#9c9280',
}

# Configure background color, default font size, as well as some tick locations
rcdefaults()
rc('figure', facecolor=palette['background'], dpi=110)
rc('axes', facecolor=palette['background'])
rc('font', size=14)
rc('xtick', bottom=False)
rc(
    'ytick', direction='inout', left=True, 
    right=True, labelleft=True, labelright=True
)
rc('axes.spines', left=True, top=False, right=True, bottom=False)
from matplotlib.dates import date2num
from pandas import date_range, Timestamp, Timedelta

fig = figure(dpi=160, figsize=(25, 12))

# 2 Axes in a single column, shared-x, top Axes is 3x the height of the bottom
gs = GridSpec(
    2, 12, height_ratios=[3, 1], hspace=.2, 
    top=.9, left=.03, right=.97, bottom=.03
)

temperature_ax = fig.add_subplot(gs[0, :])
precip_ax = fig.add_subplot(gs[1, :])
precip_ax.sharex(temperature_ax)

# Custom Locator to remove Jan-1 so grid does not overlap with left spine
#   since x-axis is shared, x-axis tick locators are also shared 
class ExcludeDayLocator(DayLocator):
    def __init__(self, *args, exclude=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.exclude = date2num(exclude)
        
    def tick_values(self, vmin, vmax):
        values = super().tick_values(vmin, vmax)
        values = [v for v in values if v not in self.exclude]
        return values

# The temperature Axes should have its ticklabels along the top
# sourcery skip: assign-if-exp, switch
temperature_ax.xaxis.set_tick_params(labeltop=True, labelbottom=False, pad=10)

# temperature Axes: y-axis tick every multiple of 10
temperature_ax.yaxis.set_major_locator(MultipleLocator(10))
temperature_ax.yaxis.set_major_formatter('{x:g}°')

# Major ticks correspond to middle of each month (used for month labels only)
# Minor ticks correspond to beginning of each month (will have grid lines)
precip_ax.xaxis.set_major_locator(DayLocator(15))
precip_ax.xaxis.set_major_formatter(DateFormatter('%B'))
precip_ax.xaxis.set_minor_locator(
    ExcludeDayLocator(1, exclude=to_datetime(['2003-01-01']))
)

# precipitation Axes: y-axis tick every multiple of 2
precip_ax.yaxis.set_major_locator(MultipleLocator(2))

# Set options for BOTH temperature and precipitation Axes:
for ax in fig.axes:
    ax.spines[['left', 'right']].set_color(palette['normal_range'])
    ax.spines[['left', 'right']].set_lw(3)
    
    # This is part matplotlib/part dirty w1orkaround to have the ticks appear
    #  on top of the spines.
    ax.spines[['left', 'right']].set_zorder(1)
    # Lines in matplotlib have some extra padding on either end by default.
    #  This removes that padding from the spines
    ax.spines[['left', 'right']].set_capstyle('butt')
    
    # I want the ticks to appear as transparent breaks in each spine,
    #   so I need to ensure that the tick is the same color as the background
    #   and are longer than the spine is wide.
    length = ax.spines['left'].get_lw() + 1
    
    # The grid on the y-axis is illusory and will only be apparent when there
    #  is data on the Axes
    ax.yaxis.grid(
        which='major', lw=1, color=palette['background'], ls='-', clip_on=False
    )
    ax.yaxis.set_tick_params(
        length=length, width=2, color=palette['background']
    )
    
    # A non-illusory vertical grid 
    ax.xaxis.grid(
        which='minor', color='gray', linestyle='dotted', linewidth=1, alpha=.8
    )

# Lastly, I copy Tufte's x & y limits for a more identical looking reproduction
temperature_ax.set_ylim(-20, 110)
temperature_ax.yaxis.set_major_locator(MultipleLocator(10))
temperature_ax.yaxis.set_major_formatter('{x:g}°')

precip_ax.set_xlim(nyc_2003.index.min(), nyc_2003.index.max())
precip_ax.set_ylim(0, 10)

# since the temperature and percipitation Axes share an x-axis, changing the 
#  limits on one will change the limits on the other.
precip_ax.set_xlim(nyc_2003.index.min(), nyc_2003.index.max())

setp(temperature_ax.get_xticklabels(), weight='bold')
setp(precip_ax.get_xticklabels(), weight='bold')

# st.pyplot(fig)

temperature_legend_ax = temperature_ax.inset_axes([.5, 0, .01, .3])
temperature_legend_ax.set_anchor('SC')
temperature_legend_ax.axis('off')

for ax in [temperature_ax, temperature_legend_ax]:
    ax.bar(
        plot_data.index,
        plot_data['historical_max'] - plot_data['historical_min'],
        bottom=plot_data['historical_min'],
        color=palette['record_range'], edgecolor=palette['record_range'],
    )

    ax.bar(
        plot_data.index,
        plot_data['normal_max'] - plot_data['normal_min'],
        bottom=plot_data['normal_min'],
        color=palette['normal_range'], edgecolor=palette['normal_range'],
    )

    ax.bar(
        plot_data.index,
        height=plot_data['temperature_max'] - plot_data['temperature_min'], 
        bottom=plot_data['temperature_min'],
        color=palette['daily_range'], linewidth=0, width=.9,
        zorder=1.1 # daily data should sit on top of grid lines
    )

# limit the `temperature_legend_ax` to just show part of a day of data
legend_date = Timestamp('2003-07-07')
temperature_legend_ax.set_xlim(legend_date, legend_date + Timedelta(1, 'H'))
temperature_legend_ax.set_ylim(50, 100) # zoom in y-axis

# st.pyplot(fig)

temperature_legend_ax.annotate(
    'RECORD HIGH',
    xy=(0, plot_data.loc[legend_date, 'historical_max']), 
    xycoords=temperature_legend_ax.get_yaxis_transform(),
    xytext=(-5, 0),
    textcoords='offset points', 
    ha='right', va='center',
)
temperature_legend_ax.annotate(
    'RECORD LOW',
    xy=(0, plot_data.loc[legend_date, 'historical_min']), 
    xycoords=temperature_legend_ax.get_yaxis_transform(),
    xytext=(-5, 0),
    textcoords='offset points', 
    ha='right', va='center',
)
temperature_legend_ax.annotate(
    'ACTUAL HIGH',
    xy=(1, plot_data.loc[legend_date, 'temperature_max']), 
    xycoords=temperature_legend_ax.get_yaxis_transform(),
    xytext=(10, 0),
    textcoords='offset points',
    arrowprops={'arrowstyle': '-', 'linewidth': 2},
    ha='left', va='center',
)
temperature_legend_ax.annotate(
    'ACTUAL LOW',
    xy=(1, plot_data.loc[legend_date, 'temperature_min']), 
    xycoords=temperature_legend_ax.get_yaxis_transform(),
    xytext=(10, 0),
    arrowprops={'arrowstyle': '-', 'linewidth': 2},
    textcoords='offset points', 
    ha='left', va='center',
)

# st.pyplot(fig)

from matplotlib.patches import ConnectionPatch

offset_trans = offset_copy(
    temperature_legend_ax.get_yaxis_transform(), 
    x=-10, y=0, fig=fig,
    units='points'
)
for point in ['normal_min', 'normal_max']:
    conn = ConnectionPatch(
        (0, plot_data.loc[legend_date, ['normal_min', 'normal_max']].mean()),
        (0, plot_data.loc[legend_date, point]),
        coordsB=temperature_legend_ax.get_yaxis_transform(),
        coordsA=offset_trans,
        lw=2, color='k', zorder=20,
        connectionstyle='angle',
    )

    temperature_legend_ax.add_artist(conn)

temperature_legend_ax.annotate(
    'NORMAL RANGE', 
    xy=(0, plot_data.loc[legend_date, ['normal_min', 'normal_max']].mean()),
    xycoords=offset_trans,
    xytext=(-5, 0), textcoords='offset points',
    ha='right', va='center'
)


# st.pyplot(fig)

from matplotlib.transforms import IdentityTransform
from matplotlib.offsetbox import TextArea, AnchoredOffsetbox, VPacker, HPacker

bbox = temperature_ax.get_tightbbox()
ytrans = offset_copy(temperature_ax.transAxes, x=0, y=50, fig=fig, units='points')
transform = blended_transform_factory(IdentityTransform(), ytrans)
fig.text(
    s='New York City’s Weather in 2003', x=bbox.x0, y=1, 
    ha='left', va='bottom', transform=transform,
    weight='bold', size='xx-large'
)

annual_avg_temps = (
    nyc_historical.filter(like='temperature')
    .resample('YS')
    .mean().mean(axis='columns') # no NumPy-like axis=None
)
avg_year_temp = annual_avg_temps.at['2003-01-01']
prev_cold_year = annual_avg_temps.loc[
    lambda s: (s <= avg_year_temp) & (s.index.year < 2003)
].index[-1]

temperature_annot = VPacker(
    children=[
        TextArea('Temperature', textprops={'weight': 'bold', 'size': 'large'}), 
        TextArea('\n'.join([
            'Bars represent range between the daily high',
            'and low. Average temperature for the year was',
            f'{avg_year_temp:.1f}°, making 2003 the coldest year since {prev_cold_year:%Y}',
            ])
        )
    ], pad=0, sep=5
)

p = AnchoredOffsetbox(
    child=temperature_annot, loc='upper left',
    pad=0, bbox_to_anchor=(0, 1), borderpad=0,
    bbox_transform=(
        offset_copy(temperature_ax.transAxes, x=20, y=0, units='points', fig=fig)
    )
)
p.patch.set(facecolor=palette['background'], ec='none')
temperature_ax.add_artist(p)

# st.pyplot(fig)

grid_masks = [
    ('2003-02-01', 'above'), ('2003-03-01', 'above'), ('2003-07-01', 'below')
]

low, high = temperature_ax.get_ylim()

xs, ymins, ymaxes = [], [], []
for date, pos in grid_masks:
    xs.append(Timestamp(date))
    if pos == 'above':
        ymins.append(plot_data.loc[date, 'historical_max'])
        ymaxes.append(high)
    elif pos == 'below':
        ymins.append(low)
        ymaxes.append(plot_data.loc[date, 'historical_min'])
        
temperature_ax.vlines(xs, ymins, ymaxes, lw=2, color=palette['background'])

# st.pyplot(fig)

daily_temperature_records = (
    nyc_historical
    .groupby(nyc_historical.index.dayofyear)
    .agg({
        'temperature_max': ['idxmax', 'max'],
        'temperature_min': ['idxmin', 'min']}
    )
)

trans = offset_copy(temperature_ax.get_xaxis_transform(), x=5, y=0, fig=fig, units='points')
max_temp_annot_kwargs = {'va': 'top', 'y': .98, 's': 'RECORD HIGH {}°'}
temp_high_records_in_2003 = (
    daily_temperature_records['temperature_max']
    .loc[lambda d: d['idxmax'].dt.year.eq(2003)]
)
for _, row in temp_high_records_in_2003.iterrows():
    text = temperature_ax.text(
        x=date2num(row['idxmax']), y=.98, s=f'RECORD HIGH {row["max"]:.0f}°', 
        transform=trans, va='top', ha='left',
        fontsize='small'
    )

    conn = ConnectionPatch(
        xyA=(date2num(row['idxmax']), row['max']), 
        coordsA=temperature_ax.transData,
        xyB=text.get_position(),
        coordsB=temperature_ax.get_xaxis_transform(),
        color='gray', zorder=5
    )
    temperature_ax.add_artist(conn)

temp_low_records_in_2003 = (
    daily_temperature_records['temperature_min']
    .loc[lambda d: d['idxmin'].dt.year.eq(2003)]
)

for _, row in temp_low_records_in_2003.iterrows():
    text = temperature_ax.text(
        x=date2num(row['idxmin']), y=.05, s=f'RECORD LOW {row["min"]:.0f}°', 
        transform=trans, va='bottom', ha='left',
        fontsize='small'
    )

    conn = ConnectionPatch(
        xyA=(date2num(row['idxmin']), row['min']), 
        coordsA=temperature_ax.transData, 
        xyB=text.get_position(),
        coordsB=temperature_ax.get_xaxis_transform(),
        color='gray', zorder=5
    )
    temperature_ax.add_artist(conn)
    
# st.pyplot(fig)

from pandas.tseries.offsets import DateOffset

null_idx = plot_data.index.union(
    date_range('2003-01-01 00:00:01', freq='M', periods=12)
)

# Use `reindex` to introduce `null` values between each of our months
precip_plotting = plot_data['monthly_cumul_precip'].reindex(null_idx)

precip_ax.plot(
    precip_plotting.index, precip_plotting, lw=5, solid_capstyle='butt'
)
precip_ax.fill_between(
    precip_plotting.index, precip_plotting, color=palette['record_range']
)

# Draw horizontal lines representing the 
#   average total monthly precip across all years
precip_monthly = (
    nyc_2003.resample('MS')['precipitation'].sum()
    .to_frame('actual')
    .assign(normal=(
        nyc_historical.resample('MS')['precipitation'].sum()
        .pipe(lambda s: s.groupby(s.index.month).mean())
        .pipe(lambda s:
              s.set_axis(
                  [Timestamp(year=2003, month=i, day=1) for i in s.index]
              )
        ))
    )
)

precip_ax.hlines(
    precip_monthly['normal'],
    xmin=precip_monthly.index,
    xmax=precip_monthly.index + DateOffset(months=1)
)

# st.pyplot(fig)

precip_annual = (
    nyc_historical.resample('YS')['precipitation'].agg(['sum', 'count'])
    .query('count > 300')
    .assign(rank=lambda d: d['sum'].rank(ascending=False))
)

precip_annots = {
    'total': precip_annual.loc['2003-01-01', 'sum'],
    'rank': precip_annual.loc['2003-01-01', 'rank'],
    'normal_diff': (
        precip_annual.loc['2003-01-01', 'sum'] -
        precip_annual.loc[:'2003-01-01', 'sum'].mean()
    )
}

precip_annot_trans = offset_copy(precip_ax.transAxes, x=20, y=30, units='points', fig=fig)
precip_annot = HPacker(
    children=[
        TextArea('Precipitation', textprops={'weight': 'bold', 'size': 'large'}),
        TextArea(' '.join([
            'Cumulative monthly precipitation in inches compared with normal',
            'monthly precipitation. Total precipitation in 2003 was',
            '{total:.2f} inches, {normal_diff:.2f} more than normal, which',
            'makes the year the {rank:.0f}th wettest on record',
            ]).format(**precip_annots),
        )
    ], pad=0, sep=9,
)
p = AnchoredOffsetbox(child=precip_annot, loc='upper left', pad=0, bbox_to_anchor=(0, 1), borderpad=0, bbox_transform=precip_annot_trans, frameon=False)
temperature_ax.add_artist(p)

# st.pyplot(fig)

monthly_records = (
    nyc_historical.resample('M')[['precipitation', 'snowfall']].sum()
    .pipe(lambda s: s.groupby(s.index.month).rank(ascending=False))
    .loc['2003']
    .loc[lambda d: d.lt(15).any(axis=1)]
)

monthly_records

monthly_annots = [
    {'x': '2003-02-15', 's': '6th snowiest Feb.', 'y': .95},
    {'x': '2003-04-15', 's': '4th snowiest April.', 'y': .95},
    {'x': '2003-06-15', 's': 'Wettest June\non record', 'y': .05, 'va': 'bottom'},
    {'x': '2003-12-15', 's': '3rd snowiest Dec.', 'y': .95},
]
monthly_annots_defaults = {
    'ha': 'center', 'va': 'top', 
    'transform': precip_ax.get_xaxis_transform(), 
    'fontsize': 'small'
}


for m in monthly_annots:
    m['x'] = to_datetime(m['x'])
    m = monthly_annots_defaults | m
    precip_ax.text(**m)
    
# st.pyplot(fig)

from calendar import month_name

precip_annot_defaults = {
    'normal': {'xytext': (1,  3), 'ha': 'left', 'va': 'bottom', 'fontsize': 'small', 'style': 'italic'},
    'actual': {'xytext': (-1, 3), 'ha': 'right', 'va': 'bottom', 'fontsize': 'small'}
}

monthly_options = {
    'April': {'actual': {'va': 'top', 'xytext': (-2, -18)}},
    'June': {
        'normal': {'ha': 'right', 'xytext': (-2, 3)},
        'actual': {'va': 'top', 'xytext': (-2, -5)}
    },
    'August': {'normal': {'ha': 'right', 'xytext': (-5, -5), 'va': 'top'}}
}

for m in month_name[1:]:
    opts = monthly_options.get(m, {})
    opts['normal'] = precip_annot_defaults['normal'] | opts.get('normal', {})
    opts['actual'] = precip_annot_defaults['actual'] | opts.get('actual', {})
    monthly_options[m] = opts


for i, (date, row) in enumerate(precip_monthly.iterrows()):
    if i == 0:
        normal_prefix, actual_prefix = 'NORMAL\n', 'ACTUAL '
    else:
        normal_prefix, actual_prefix = '', ''
        
    left, right = date + DateOffset(days=1, minutes=-1), date + DateOffset(months=1, days=-1)
    options = (
        monthly_options.get(date.strftime('%B'))
    )
    
    x = left if options['normal']['ha'] == 'left' else right
    precip_ax.annotate(
        f"{normal_prefix}{row['normal']:.2f}", 
        xy=(x, row['normal']), textcoords='offset points',
        **options['normal']
    )    
    
    x = left if options['actual']['ha'] == 'left' else right
    precip_ax.annotate(
        f"{actual_prefix}{row['actual']:.2f}", 
        xy=(x, row['actual']), textcoords='offset points',
        **options['actual']
    )

# st.pyplot(fig)

daily_records = (
    nyc_historical.groupby(nyc_historical.index.dayofyear)
    [['precipitation', 'snowfall']].rank(ascending=False)
    .loc['2003']
    .mask(lambda d: d > 1)
    .stack()
)

daily_records

daily_annots = [
    {'date': '2003-02-17', 'col': 'snowfall',      'text': 'RECORD\nSNOWFALLS\n{}', 'xytext': (-65, 40)},
    {'date': '2003-02-22', 'col': 'precipitation', 'text': 'RECORD\nRAINFALL\n{}',  'xytext': (-25, 40)},
    {'date': '2003-04-07', 'col': 'snowfall',      'text': 'RECORD\nSNOWFALLS {}',  'xytext': (-15, 70)},
    {'date': '2003-06-04', 'col': 'precipitation', 'text': 'RECORD\nRAINFALL\n{}',  'xytext': (0, 40)},
    {'date': '2003-07-22', 'col': 'precipitation', 'text': 'RECORD\nRAINFALL\n{}',  'xytext': (-20, 40)},
    {'date': '2003-08-04', 'col': 'precipitation', 'text': 'RECORD\nRAINFALL {}',   'xytext': (-10, 50)},
    {'date': '2003-10-27', 'col': 'precipitation', 'text': 'RECORD\nRAINFALL {}',   'xytext': (-80, 65)},
    {'date': '2003-11-19', 'col': 'precipitation', 'text': 'RECORD\nRAINFALL {}',   'xytext': (-50, 60)},
    {'date': '2003-12-06', 'col': 'snowfall', 'text': 'RECORD\nSNOWFALLS\n{}',      'xytext': (-10, 60)},
    # Too many annotations for December to fit in at once
    # {'date': '2003-12-11', 'col': 'precipitation', 'text': 'RECORD\nRAINFALL {}'},
    # {'date': '2003-12-14', 'col': 'snowfall', 'text': 'RECORD\nSNOWFALLS {}', 'xytext': (0, 60)},
    {'date': '2003-12-24', 'col': 'precipitation', 'text': 'RECORD\nRAINFALL\n{}',  'xytext': (-20, -65)},
]
daily_annots_defaults = {
    'xytext': (0, 0), 'textcoords': 'offset points', 
    'fontsize': 'x-small', 'ha': 'left', 'va': 'bottom', 
    'arrowprops': {
        'arrowstyle': '-', 'connectionstyle': 'angle', 'shrinkB': 0
    }
}

for d in daily_annots:
    date = to_datetime(d['date'])
    y = plot_data.loc[date, 'monthly_cumul_precip']
    value = nyc_2003.loc[date, d['col']]
    d['xy'] = (date, y)
    d['text'] = d['text'].format(f'{value:.2f} in')
    d.pop('date')
    d.pop('col')
    
    d = daily_annots_defaults | d
    precip_ax.annotate(**d)
    
# fig.savefig('matplotlib/tufte_2003_matplotlib.png')
fig.savefig('C:/Users/Darragh/Documents/Python/dutc/tufte_2003_matplotlib.png')
st.pyplot(fig)

