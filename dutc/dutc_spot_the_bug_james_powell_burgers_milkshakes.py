import pandas as pd
import numpy as np
from numpy import unique,where,array,tile
import streamlit as st
from pandas import DataFrame,MultiIndex,period_range,IndexSlice, Series,merge,concat, get_dummies, NA, Timestamp, to_timedelta, concat as pd_concat, CategoricalIndex,date_range
from string import ascii_lowercase
from collections import Counter
from itertools import islice, groupby, pairwise, cycle, tee, zip_longest, chain, repeat, takewhile, product
from numpy.random import default_rng
from io import StringIO
from textwrap import dedent
from csv import reader
from collections import deque

st.set_page_config(layout="wide")

with st.expander("James Powell Pandas"):
    # https://www.youtube.com/watch?v=J8dJAekOkSU
    monthly_sales = DataFrame({
    'store': ['New York', 'New York', 'New York','San Francisco','San Francisco','San Francisco','Chicago','Chicago','Chicago'],
    'item': [ 'milkshake' ,  'french fry' ,  'hamburger','milkshake' ,  'french fry' ,  'hamburger','milkshake' ,  'french fry' ,  'hamburger' ],
    'quantity': [ 100,110,120,200,210,220,300,310,320 ],
    }).set_index(['store','item'])
    forecast = DataFrame({
    'quarter': ['2025Q1', '2025Q2', '2025Q3', '2025Q4'],
    'value': [ 1.01 ,  1.02 ,  1.03 ,  1.04 ],
    }).set_index('quarter')
    st.write('sales', monthly_sales,'forecast',forecast)

    index = MultiIndex.from_product([*monthly_sales.index.levels,forecast.index])
    st.write('monthly sales level',monthly_sales.index.levels)
    st.write('monthly sales index',monthly_sales.index)
    st.write('index',index)
    with st.echo():
        st.write('reindex',forecast.reindex(index,level=forecast.index.name))
        # st.write('level name',forecast.index.name)
        st.write('monthly sales', monthly_sales)
        st.write(monthly_sales * forecast.reindex(index,level=forecast.index.name))

    rng=default_rng(0)
    df = DataFrame(
        index=(idx := MultiIndex.from_product([
            period_range('2000-01-01', periods=8,freq='Q'),
            ['New York','Chicago','San Francisco'],
        ],names=['quarter','location'])),
        data={
            'revenue': +rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
            'expenses': -rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
            'manager': tile(['Alice Adams','Bob Brown','Charlie Cook','Dana Daniels'], len(idx)//4),
        }
    ).pipe(lambda df:
           merge(
               df,
               Series(
                   index=(idx := df['manager'].unique()),
                   data=rng.choice([True,False],p=[.75,.25],size=len(idx)),
                   name='franchise',
               ),
               left_on='manager',
               right_index=True,
        )
    )

    st.write('mad df',df.sort_index())
    st.write(df[~df['franchise']])

    # test_df = DataFrame(
    #     index=(idx := MultiIndex.from_product([
    #         period_range('2000-01-01', periods=8,freq='D'),
    #         ['New York','Chicago','San Francisco'],
    #     ],names=['quarter','location'])),
    #     data={
    #         'revenue': +rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
    #         'expenses': -rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
    #         'manager': tile(['Alice Adams','Bob Brown','Charlie Cook'], len(idx)//3),
    #     }
    # )

    sales = DataFrame(
        index=(idx := MultiIndex.from_product([
            date_range('2000-01-01', periods=20,freq='D'),
            ['New York','Chicago','San Francisco','Houston'],
            ['burger','fries','combo','milkshakes'],
        ],names=['date','location','item'])),
        data={
            'quantity': +rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
            
        }
    )
    

    sales_test = DataFrame(
        index=(idx := MultiIndex.from_product([
            date_range('2000-01-01', periods=20,freq='D'),
            ['New York','Chicago','San Francisco'],
            # tile(['burger','fries','combo','milkshakes'], len(idx)//4),
        ],names=['date','location'])),
        data={
            'quantity': +rng.normal(loc=10_000,scale=1_000,size=len(idx)).round(-1),
            'item': tile(['burger','fries','combo','milkshakes'], len(idx)//4),
            
        }
    )
    # .set_index(['date','location','item'])
    # st.write('test sales', sales_test,'PUT in index')

    # st.write('sales',sales,'unstack sales',sales.unstack('item','fill_value=0'))
    # st.write('test sales',sales_test)


    date_rng = pd.date_range('2000-01-01', periods=20, freq='D')
    locations = ['New York', 'Chicago', 'San Francisco']
    items = ['hamburger', 'french fry', 'combo', 'milkshake']
    idx = pd.MultiIndex.from_product([date_rng, locations, items], names=['date', 'location', 'item'])
    np.random.seed(42)
    quantities = np.random.normal(loc=10_000, scale=1_000, size=len(idx)).round(-1)
    sales = pd.DataFrame({'quantity': quantities}, index=idx)
    combo_idx = sales.index.get_level_values('item') == 'combo'
    combo_rows = sales[combo_idx]
    random_combo_idx = np.random.choice(combo_rows.index, size=int(len(combo_rows) * 0.3), replace=False)
    # Set the 'quantity' of the selected random combo rows to zero
    sales.loc[random_combo_idx, 'quantity'] = 0
    # Drop some random dates to simulate missing dates
    dates_to_remove = np.random.choice(date_rng, size=3, replace=False)
    df = sales.drop(index=dates_to_remove, level='date')
    st.write('update from chat gpt where we have some combos=0 and some missing dates',df,'unstack on this',df.unstack('item','fill_value=0'))
    
    unstack_sales=(df
             .unstack('item','fill_value=0')
             .pipe(lambda df: df
                   .reindex(
                       MultiIndex.from_product([
                           date_range((dts := df.index.get_level_values('date')).min(), dts.max(), freq='D'),
                           df.index.get_level_values('location').unique(),
                        ], names=df.index.names),
                        fill_value=0,
                   )
                   ).pipe(lambda df: df.droplevel(0, axis=1))
             )
    # st.write('this is the clean up operations',unstack_sales)
    # unstack_sales.columns = unstack_sales.columns.droplevel(0) # for some reason when I do unstack i have to drop down a level for column names, james powell
    # didn't seem to get this error
    # fixed by asking chat gpt how i could fit the above into it so added this to james code
    # .pipe(lambda df: df.droplevel(0, axis=1))
    st.write(unstack_sales.loc[IndexSlice[:,['Chicago','New York'],   :]])
    st.write(unstack_sales.columns)

    st.write(df
             .unstack('item','fill_value=0')
             .pipe(lambda df: df
                   .reindex(
                       MultiIndex.from_product([
                           date_range((dts := df.index.get_level_values('date')).min(), dts.max(), freq='D'),
                           df.index.get_level_values('location').unique(),
                        ], names=df.index.names),
                        fill_value=0,
                   )
                   ).pipe(lambda df: df.droplevel(0, axis=1)).pipe(lambda df:
                          concat([
                              concat([
                                  df.loc[IndexSlice[:,['Chicago','New York'],   :]][['french fry','hamburger','milkshake',]],
                                  df.loc[IndexSlice[:,['San Francisco'],        :]][['french fry','hamburger',            ]].assign(milkshake=0),
                                  ],axis='index').sort_index()
                              .rename(lambda x: ('combo',x), axis='columns'),
                              df[['french fry','hamburger','milkshake']]
                              .rename(lambda x: ('solo', x), axis='columns'),
                          ],axis='columns')
                          .pipe(lambda df: df.set_axis(MultiIndex.from_tuples(df.columns),axis='columns'))
                          .rename_axis(['order type', 'item'], axis='columns')
            )
    
    
                          .sort_index()
    # )
                          .groupby('item',axis='columns').sum()
             )
                    
    # st.write('full output', full_output)
    # st.write('index length',len(idx := MultiIndex.from_product([
    #         period_range('2000-01-01', periods=8,freq='D'),
    #         ['New York','Chicago','San Francisco'],
    #     ],names=['quarter','location'])))
    # st.write('index',DataFrame(index=(idx := MultiIndex.from_product([
    #         period_range('2000-01-01', periods=8,freq='D'),
    #         ['New York','Chicago','San Francisco'],
    #     ],names=['quarter','location']))))
    
    # st.write('test df', test_df.sort_index())
    # st.write(len(idx))
    # st.write(len(idx)//7)
    # st.write(tile(['Alice Adams','Bob Brown','Charlie Cook','Dana Daniels'], len(idx)//4))
    # st.write(tile(['Alice Adams','Bob Brown','Charlie Cook','Dana Daniels'], len(idx)//12))

with st.expander('pure python alternative'):
    from datetime import datetime, timedelta
    import random

    # Define the date range
    start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
    date_rng = [start_date + timedelta(days=x) for x in range(20)]

    # Define locations and items
    locations = ['New York', 'Chicago', 'San Francisco']
    items = ['hamburger', 'french fry', 'combo', 'milkshake']

    # Generate the index (cartesian product)
    index = [(date, location, item) for date in date_rng for location in locations for item in items]

    index_itertools = list(product(date_rng, locations, items))
    # st.write('index using tuple',index,'index using itertoools', index_itertools)

    # Helper function to convert a tuple to a JSON-compatible string key
    def tuple_to_str(tup):
        date, location, item = tup
        return f"{date.strftime('%Y-%m-%d')}|{location}|{item}"

    # Convert the index tuples to strings
    index_str_keys = [tuple_to_str(idx) for idx in index]
    # st.write('index string keys', index_str_keys)

    # Set seed for reproducibility
    random.seed(42)

    # Generate random sales quantities
    quantities = [round(random.gauss(10000, 1000), -1) for _ in index]

    # Create the sales dictionary with string keys
    sales = {idx_str: quantity for idx_str, quantity in zip(index_str_keys, quantities)}
    sales_normal = {idx: quantity for idx, quantity in zip(index, quantities)}
    # st.write('sales',sales,'tuple sales', sales_normal)

    # Example: Randomly Set 'combo' Quantities to Zero
    combo_rows = [key for key in sales.keys() if 'combo' in key]

    # Randomly select 30% of combo rows
    random_combo_idx = random.sample(combo_rows, k=int(len(combo_rows) * 0.3))

    # Set selected 'combo' quantities to zero
    for key in random_combo_idx:
        sales[key] = 0

    # Randomly select 3 dates to remove
    dates_to_remove = random.sample(date_rng, k=3)

    # Convert dates to string format for matching keys
    dates_to_remove_str = [date.strftime('%Y-%m-%d') for date in dates_to_remove]

    # Remove entries with these dates
    sales = {key: quantity for key, quantity in sales.items() if key.split('|')[0] not in dates_to_remove_str}

    # Fill in missing dates with zeros
    all_dates = [start_date + timedelta(days=x) for x in range((max(date_rng) - min(date_rng)).days + 1)]
    all_dates_str = [date.strftime('%Y-%m-%d') for date in all_dates]
    all_locations = locations
    filled_sales = {}

    # Ensure that missing dates and combinations have zeros
    for date in all_dates_str:
        for location in all_locations:
            for item in items:
                key = f"{date}|{location}|{item}"
                if key in sales:
                    filled_sales[key] = sales[key]
                else:
                    filled_sales[key] = 0

    # Aggregate data into combo and solo groups
    output = {}

    for key, quantity in filled_sales.items():
        date_str, location, item = key.split('|')
        
        if location in ['Chicago', 'New York']:
            if item in ['french fry', 'hamburger', 'milkshake']:
                combo_key = f"{date_str}|{location}|combo|{item}"
                output[combo_key] = output.get(combo_key, 0) + quantity
        elif location == 'San Francisco':
            if item in ['french fry', 'hamburger']:
                combo_key = f"{date_str}|{location}|combo|{item}"
                output[combo_key] = output.get(combo_key, 0) + quantity
            elif item == 'milkshake':
                combo_key = f"{date_str}|{location}|combo|{item}"
                output[combo_key] = 0

        solo_key = f"{date_str}|{location}|solo|{item}"
        output[solo_key] = output.get(solo_key, 0) + quantity

        # Group by item and sum the quantities
        final_output = {}
        for key, value in output.items():
            date_str, location, order_type, item = key.split("|")
            if item not in final_output:
                final_output[item] = {}
            order_location_key = f"{date_str}|{location}|{order_type}"
            if order_location_key not in final_output[item]:
                final_output[item][order_location_key] = 0
            final_output[item][order_location_key] += value
        
    # st.write(final_output)

# sourcery skip: assign-if-exp, default-get, dict-comprehension, identity-comprehension, merge-duplicate-blocks, remove-dict-keys, remove-pass-elif, remove-redundant-if
# sourcery skip: assign-if-exp, default-get, identity-comprehension, merge-duplicate-blocks, remove-pass-elif, remove-redundant-if
with st.expander("new answe"):
    from datetime import datetime, timedelta
    import random

    # Define the date range
    start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
    date_rng = [start_date + timedelta(days=x) for x in range(20)]

    # Define locations and items
    locations = ['New York', 'Chicago', 'San Francisco']
    items = ['hamburger', 'french fry', 'combo', 'milkshake']

    # Generate the index (cartesian product)
    index = [(date, location, item) for date in date_rng for location in locations for item in items]

    # Set seed for reproducibility
    random.seed(42)

    # Generate random sales quantities
    quantities = [round(random.gauss(10000, 1000), -1) for _ in index]

    # Create the sales dictionary using tuples as keys
    sales = {idx: quantity for idx, quantity in zip(index, quantities)}

    # Example: Randomly Set 'combo' Quantities to Zero
    combo_rows = [key for key in sales if key[2] == 'combo']

    # Randomly select 30% of combo rows
    random_combo_idx = random.sample(combo_rows, k=int(len(combo_rows) * 0.3))

    # Set selected 'combo' quantities to zero
    for key in random_combo_idx:
        sales[key] = 0

    # Randomly select 3 dates to remove
    dates_to_remove = random.sample(date_rng, k=3)

    # Remove entries with these dates
    sales = {key: quantity for key, quantity in sales.items() if key[0] not in dates_to_remove}

    # Fill in missing dates with zeros
    all_dates = [start_date + timedelta(days=x) for x in range((max(date_rng) - min(date_rng)).days + 1)]
    filled_sales = {}

    # Ensure that missing dates and combinations have zeros
    for date in all_dates:
        for location in locations:
            for item in items:
                key = (date, location, item)
                filled_sales[key] = sales.get(key, 0)

    # Aggregate data into combo and solo groups
    output = {}

    for key, quantity in filled_sales.items():
        date, location, item = key
        
        if location in ['Chicago', 'New York']:
            if item in ['french fry', 'hamburger', 'milkshake']:
                combo_key = (date, location, f'combo|{item}')
                output[combo_key] = output.get(combo_key, 0) + quantity
        elif location == 'San Francisco':
            if item in ['french fry', 'hamburger']:
                combo_key = (date, location, f'combo|{item}')
                output[combo_key] = output.get(combo_key, 0) + quantity
            elif item == 'milkshake':
                combo_key = (date, location, f'combo|{item}')
                output[combo_key] = 0

        solo_key = (date, location, f'solo|{item}')
        output[solo_key] = output.get(solo_key, 0) + quantity

    # Group by item and sum the quantities
    final_output = {}
    for key, value in output.items():
        date, location, order_type_item = key
        order_type, item = order_type_item.split("|")
        
        if item not in final_output:
            final_output[item] = {}
            
        order_location_key = (date, location, order_type)
        
        if order_location_key not in final_output[item]:
            final_output[item][order_location_key] = 0
            
        final_output[item][order_location_key] += value

    # Output the final result
    # print(final_output)
    # st.write(final_output)

with st.expander('original code'):
    from datetime import datetime, timedelta
    import random

    # Define the date range
    start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
    date_rng = [start_date + timedelta(days=x) for x in range(20)]

    # Define locations and items
    locations = ['New York', 'Chicago', 'San Francisco']
    items = ['hamburger', 'french fry', 'combo', 'milkshake']

    # Generate the index (cartesian product)
    index = [(date, location, item) for date in date_rng for location in locations for item in items]

    # Set seed for reproducibility
    random.seed(42)

    # Generate random sales quantities
    quantities = [round(random.gauss(10000, 1000), -1) for _ in index]

    # Create the sales dictionary
    sales = {idx: quantity for idx, quantity in zip(index, quantities)}

    # Filter for 'combo' items
    combo_rows = [idx for idx in sales.keys() if idx[2] == 'combo']

    # Randomly select 30% of combo rows
    random_combo_idx = random.sample(combo_rows, k=int(len(combo_rows) * 0.3))

    # Set selected 'combo' quantities to zero
    for idx in random_combo_idx:
        sales[idx] = 0

    # Randomly select 3 dates to remove
    dates_to_remove = random.sample(date_rng, k=3)

    # Remove entries with these dates
    sales = {idx: quantity for idx, quantity in sales.items() if idx[0] not in dates_to_remove}

    # Fill in missing dates with zeros
    all_dates = [start_date + timedelta(days=x) for x in range((max(date_rng) - min(date_rng)).days + 1)]
    all_locations = locations
    filled_sales = {}

    for date in all_dates:
        for location in all_locations:
            for item in items:
                idx = (date, location, item)
                if idx in sales:
                    filled_sales[idx] = sales[idx]
                else:
                    filled_sales[idx] = 0

    # Aggregate data as required
    output = {}

    for idx, quantity in filled_sales.items():
        date, location, item = idx
        
        if location in ['Chicago', 'New York']:
            if item in ['french fry', 'hamburger', 'milkshake']:
                output[(date, location, 'combo', item)] = quantity
        elif location == 'San Francisco':
            if item in ['french fry', 'hamburger']:
                output[(date, location, 'combo', item)] = quantity
            elif item == 'milkshake':
                output[(date, location, 'combo', item)] = 0

        output[(date, location, 'solo', item)] = quantity

    # Group by item and sum
    final_output = {}
    for key, value in output.items():
        date, location, order_type, item = key
        if item not in final_output:
            final_output[item] = {}
        if (date, location, order_type) not in final_output[item]:
            final_output[item][(date, location, order_type)] = 0
        final_output[item][(date, location, order_type)] += value

    # Sort the final output by date and location
    sorted_final_output = {}
    for item, data in final_output.items():
        sorted_final_output[item] = dict(sorted(data.items(), key=lambda x: (x[0][0], x[0][1])))

    # Final output is in sorted_final_output
    # print(sorted_final_output)

    # Flatten the nested dictionary
    flattened_data = []

    for item, sub_dict in sorted_final_output.items():
        for key, quantity in sub_dict.items():
            date, location, order_type = key
            flattened_data.append({
                'item': item,
                'date': date.strftime('%Y-%m-%d'),  # Convert datetime to string format
                'location': location,
                'order_type': order_type,
                'quantity': quantity
            })

    st.write(pd.DataFrame(flattened_data))