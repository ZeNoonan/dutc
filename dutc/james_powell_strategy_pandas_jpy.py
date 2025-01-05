from string import ascii_lowercase
from datetime import date, time, timedelta, to_timedelta
from pandas import Series, MultiIndex, date_range, CategoricalIndex, DataFrame, IndexSlice, Index, concat
from pandas.api.extensions import register_index_accessor
from numpy import unique, array, hstack, where
from numpy.random import default_rng
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product, chain
from random import Random
import streamlit as st

st.set_page_config(layout="wide")

rng = default_rng(0)

# 1st wrong attempt
dates = [date(2020,1,1) + timedelta(days=n) for n in range(90)]
times = [time(hr, min, 0) for hr, min in product(range(9,16+1), range(0, 45+1,15))]
assets = [''.join(rng.choice([*ascii_lowercase], size=4)) for _ in range(200)]
prices = [rng.random() for _ in product(dates, times, assets)]
df= DataFrame({
    'date': dates,
    'time': times,
    'asset': assets,
    'price': prices,
})

# st.write(df)
# get 'All arrays must be of same length with the below and above

df = DataFrame({
    'date': [d for d, _, _ in product(dates, times, assets)],
    'time': [t for _, t, _ in product(dates, times, assets)],
    'asset': [a for _, _, a in product(dates, times, assets)],
    'price': [rng.random() for _ in product(dates, times, assets)],
})
st.write(df)



dates = date_range('2020-01-01','2020-03-31',  freq='15min', name='date')
tickers = unique(rng.choice([*ascii_lowercase], size=(4,200)).view('<U4').ravel())
currencies = array(['USD', 'JPY'])
assets = CategoricalIndex(hstack([tickers, currencies]), name='asset')

prices = Series(
    index = MultiIndex.from_product([
        dates, 
        assets,
    ]),
    data = (
        rng.normal(loc=100,scale=20,   size=len(assets)).clip(0,200)
       *rng.normal(loc=1,  scale=.005, size=(len(dates),len(assets))).cumprod(axis=-1)
    ).ravel().round(4),
        name='price',
)

prices.loc[IndexSlice[:, 'USD']] = 1
prices.loc[IndexSlice[:, 'JPY']] = 10
prices = prices.between_time('9:30', '16:00')

# contortion
prices = prices.iloc[
    prices
    .index
    .get_level_values('date')
    .indexer_between_time('9:30', '16:00')
]

st.write('this is prices',prices)

# end of day price

st.write('end of day price',
    prices.groupby([
    prices.index.get_level_values('date').date,
    'asset',
    ], observed=True).agg(['first','last'])
    )
    
trades = DataFrame(
    index = (idx := MultiIndex.from_product([
        dates,assets,
        range(3),
        ], names=['date','asset','trade'])),
        data = {
            'volume': (
                rng.choice([+1,-1], size=len(idx))
                *rng.integers(100, 100_000, size=len(idx)).round(-2)
            ), 
        },
).pipe(
    lambda df: df.iloc[
        df.index.get_level_values('date').indexer_between_time('9:30', '16:00')
    ]
).assign(
    price = lambda df: prices.loc[
        MultiIndex.from_arrays([
            df.index.get_level_values('date'),
            df.index.get_level_values('asset'),
        ])
    ].round(2).values
).sample(frac=0.01, random_state=rng)

st.write('this is trades',trades)

refdate = prices.index.get_level_values('date').max()

trades.assign(
    market_price = lambda df: prices.loc[
        MultiIndex.from_arrays([
            [refdate]*len(df),
            df.index.get_level_values('asset'),
        ])
    ].values,
    cash = lambda df: -df['volume']*df['price'],
    market_value = lambda df: df['market_price']*df['volume'],
    profit = lambda df: df['cash'] + df['market_value'],
)

st.write('updated trades',trades)

# 17.25 Min
# prices as DataFrame DN note
prices = DataFrame(
    index = MultiIndex.from_product([
        dates, 
        assets,
    ]),
    data = {
        'buy':(
        rng.normal(loc=100,scale=20,   size=len(assets)).clip(0,200)
       *rng.normal(loc=1,  scale=.005, size=(len(dates),len(assets))).cumprod(axis=-1)
    ).ravel(),
    },
).assign(
    sell = lambda df: df['buy'] * (1 - abs(rng.normal(loc=0, scale = 0.001, size = len(df))))
).round(2)
        
st.write('now the prices look like:', prices)

# 17.41 mins
trades.assign(
    position = lambda df: df.groupby('asset', observed = True)['volume'].cumsum(),

    market_price = lambda df: prices.loc[
        MultiIndex.from_arrays([
            [refdate]*len(df),
            df.index.get_level_values('asset'),
        ])
    ].pipe( lambda px: where(df['position'] > 0, px['sell'], px['buy'])),
    cash = lambda df: -df['volume']*df['price'],
    market_value = lambda df: df['market_price']*df['volume'],
    profit = lambda df: df['cash'] + df['market_value'],
)

st.write('trades', trades)
st.write('but market value is dependant on volume and price ie deltas (trades) versus states (positions)')

# 20.42 mins
trades = DataFrame(
    index = (idx := MultiIndex.from_product([
        dates,Index(tickers, dtype=assets.dtype),
        range(3),
        ], names=['date','asset','trade'])),
        data = 
                rng.choice([+1,-1], size=len(idx))
                *rng.integers(100, 100_000, size=len(idx)).round(-2),
                name='volume'
            ).sample(frac=0.01, random_state=rng) 

def execute(volumes, prices, slippage):
    traded_prices = prices.loc[
        MultiIndex.from_arrays([
            volumes.index.get_level_values('date').floor('min'),
            volumes.index.get_level_values('asset'),
        ])
    ].pipe(lambda df: where(volumes > 0, df['buy'], df['sell'])) * slippage
    return (
        (traded_prices*-volumes)
        .pipe(lambda s: s.set_axis(s.index._ext.updatelevel(asset=['USD']*len(volumes))))
    )

# 21.08 mins
st.write(concat[trades, execute(trades, prices)])

# 21.42 mins

Liquidation = namedtuple('Liquidation', 'trades cashflows')

def liquidate(volumes, prices, *, date=None):
    dates=[
        prices.index.get_level_values('date').max() if date is None else date
    ] * len(volumes)
    trades = (-volumes).pipe(lambda s: s.set_axis(s.index._ext.updatelevel(date=dates)))
    return Liquidation(trades = trades, cashflows = execute(trades, prices, 1))

# 21.55 mins
st.write( concat([trades, *liquidate(trades, prices)]) )

# 22.14 mins
@register_index_accessor('_ext')
@dataclass
class _ext:
    obj : Index
    def addlevel (self, **levels):
        levels = {k: v if not isinstance(v, Callable) else v(self.obj) for k,v in levels.items()}
        new_obj = self.obj.copy(deep=False)
        if not isinstance(new_obj, MultiIndex):
            new_obj = MultiIndex.from_arrays([new_obj])
        names = new_obj.names
        new_obj.names = [None] * len(names)
        return MultiIndex.from_arrays([
            *(new_obj.get_level_values(idx) for idx in range(len(names))),
            *levels.values(),
        ], names = [*names, *levels.keys()])
        
    def updatelevel(self, **levels):
        levels = {k: v if not isinstance(v, Callable) else v(self.obj) for k,v in levels.items()}
        new_obj = self.obj.copy(deep=False)
        if not isinstance(new_obj, MultiIndex):
            new_obj = MultiIndex.from_arrays([new_obj])
        names = new_obj.names
        new_obj.names = [None] * len(names)
        return MultiIndex.from_arrays([
            levels[n] if n in levels else new_obj.get_level_values(idx) for idx, n in enumerate(names)
        ], names = names)
    

# 22.22 mins
st.write('positions',
         concat([
             trades,
             execute(trades, prices, 1),
         ]).groupby('asset').sum()
         )

st.write('profit and loss',
         concat([
             trades,
             execute(trades, prices, 1),
             * liquidate(trades, prices),
         ]).groupby('asset').sum()
         )

st.write('p&l',
         liquidate(trades, prices).cashflows.groupby('asset').sum()
)

liq = liquidate(trades, prices)
st.write('p&l',
        liq.cashflows.pipe(
            lambda s: s
            .set_axis(s.index._ext.addlevel(traded_asset=liq.trades.index.get_level_values('asset')))
        ).groupby(['traded_asset', 'asset'],observed=True).sum().unstack('asset')
)
# min 28.29 mins
def execute(volumes, prices):
    volumes = [volumes] if isinstance(volumes, Series) else volumes
    traded_prices = [
        prices.loc[
            MultiIndex.from_arrays([
                vol.index.get_level_values('date').floor('min'),
                vol.index.get_level_values('asset'),
            ])
        ].pipe(lambda df: where(volumes > 0, df['buy'], df['sell']))
        for vol in volumes
    ]
    return [
        (px * -vol)
        .pipe(lambda s: s.set_axis(s.index._ext.updatelevel(asset=['USD']*len(volumes))))
              for px, vol in zip(volumes, traded_prices)
    ]

Liquidation = namedtuple('Liquidation', 'trades cashflows')
# def liquidate(volumes: Series, prices: DataFame, *, date=None) -> Liquidation:

class Liquidation(namedtuple('Liquidation','trades cashflows')):
    def __iter__(self):
        return chain.from_iterable(self)
def liquidate(volumes: list[Series], prices: DataFrame,*, date=None)->Liquidation:
    ...

def liquidate(volumes, prices, *, date=None):
    dates = [
        [prices.index.get_level_values('date').max() if date is None else date] * len(vol)
        for vol in volumes
    ]
    trades = [
        (-vol).pipe(lambda s: s.set_axis(s.index._ext.updatelevel(date=dt)))
        for dt, vol in zip(dates, volumes)
    ]
    return Liquidation(trades=trades, cashflows=[execute(tr, prices) for t in trades])

# 29.07 min
Liquidation = namedtuple('Liquidation', 'trades cashflows')
class Market:
    def execute(volumes: list[Series])->list[Series]:
        ...
    def liquidate(volumes: list[Series],*, date=None, keep_asset=False)->Liquidation:
        ...

# 30.04 min
def simulate(initial_trades, dates, strategy, market):
    trades = [initial_trades]
    for dt in dates:
        trades.extend(
            strategy(dt, trades, market)
        )
    return trades

def strategy(dt, trades, market):
    returns = concat(
        liquidate([
            market.sample_portfolio.add_level(date=dt - to_timedelta('90d')),
            market.sample_portfolio.add_level(date=dt),
        ], keep_asset=True).cashflows
    ).sort_index().groupby('asset').agg(lambda s: s.iloc[-1] / s.iloc[0] -1).nlargest(20)

    new_trades = ...

    return [
        *trades, *liquidate(trades),
        *new_trades, *liquidate(new_trades)
    ]

if __name__ == '__main__':
    initial_trades = [Series(
        index=Index(['USD'], dtype=assets.dtype),
        data=100_000,
    )]
    trades = simulate(initial_trades, date_range('2020-01-01', '2020-12-31'), strategy, market)
                      