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
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(layout="wide")

with st.expander('Test my understanding of the yield by using the mortgage example'):
    from datetime import date
    import numpy as np
    import numpy_financial as np_fin
    from collections import OrderedDict
    from dateutil.relativedelta import *

    # https://pbpython.com/amortization-model-revised.html
    def amortize(principal, interest_rate, years, addl_principal=0, annual_payments=12, start_date=date.today()):

        pmt = -round(np_fin.pmt(interest_rate/annual_payments, years*annual_payments, principal), 2)
        # initialize the variables to keep track of the periods and running balances
        p = 1
        beg_balance = principal
        end_balance = principal

        while end_balance > 0:

            # Recalculate the interest based on the current balance
            interest = round(((interest_rate/annual_payments) * beg_balance), 2)

            # Determine payment based on whether or not this period will pay off the loan
            pmt = min(pmt, beg_balance + interest)
            principal = pmt - interest

            # Ensure additional payment gets adjusted if the loan is being paid off
            addl_principal = min(addl_principal, beg_balance - principal)
            end_balance = beg_balance - (principal + addl_principal)

            yield OrderedDict([('Month',start_date),
                            ('Period', p),
                            ('Begin Balance', beg_balance),
                            ('Payment', pmt),
                            ('Principal', principal),
                            ('Interest', interest),
                            ('Additional_Payment', addl_principal),
                            ('End Balance', end_balance)])

            # Increment the counter, balance and date
            p += 1
            start_date += relativedelta(months=1)
            beg_balance = end_balance

    schedule = pd.DataFrame(amortize(700000, .04, 30, addl_principal=200, start_date=date(2016, 1,1)))
    st.write(schedule.head(),schedule.tail())

with st.expander(''):
    pass
    def simple_df(op_bal=0,salary=200):
        cl_bal = 0
        n=0
        payment = yield
        cl_bal=op_bal - payment + salary
        n=n+1
        yield OrderedDict( [('Opening Balance', op_bal) , ('Payments', payment) , ('Closing Balance', cl_bal), ('Period', n) ])
        op_bal = cl_bal

    
    result=simple_df()
    
with st.expander('Claude example'):
    from collections import OrderedDict
    from datetime import date
    from dateutil.relativedelta import relativedelta

    # https://claude.ai/chat/06498b4b-9ed3-41c3-b1eb-f9a4141164ff

    def demo_generator():
        """
        Demonstrates that generator doesn't store previous values by default.
        """
        period = 1
        balance = 1000
        
        while balance > 0:
            # Create and immediately yield current data
            yield OrderedDict([
                ('Period', period),
                ('Balance', balance)
            ])
            
            # Update for next iteration
            period += 1
            balance -= 100
            
    # Let's see what happens when we use the generator different ways:


    def test_gen():
        yield 1
        yield 2
        yield 3

    testing_instance=test_gen()
    st.write('test', list(testing_instance))
    st.write("1. Using generator directly (no storage):")
    gen = demo_generator()
    df_wrap=demo_generator()
    list_wrap=demo_generator()
    st.write('what if i wrap it in a dataframe',pd.DataFrame(df_wrap))
    st.write('what if i wrap it in a dataframe',list(list_wrap))
    st.write('does this mean that putting the generator in a dataframe is like unpacking the generator say similar to using *')
    for _ in range(3):
        payment = next(gen)
        # st.write(f"Current payment: {dict(payment)}")
        st.write(f"Current payment: {payment}") # DN Adjustment to see what happens: interesting looks like Ordered Dict is a list of tuples??
        st.write("Can we access previous payments? No - they're gone!")
    st.write()

    st.write("2. Manually collecting into a list:")
    all_payments = []  # We create the storage
    gen = demo_generator()
    for _ in range(3):
        payment = next(gen)
        all_payments.append(payment)  # We explicitly save each payment
        st.write(f"Current payment: {dict(payment)}")
        st.write(f"Stored payments: {[dict(p) for p in all_payments]}")
    st.write()

    st.write("3. Using list comprehension to collect all at once:")
    all_at_once = list(demo_generator())  # Collects all yields into a list
    st.write(f"First 3 payments: {[dict(p) for p in all_at_once[:3]]}")
    st.write()

    st.write("4. How pandas does it behind the scenes:")
    import pandas as pd

    # Pandas essentially does this:
    data_for_df = list(demo_generator())  # First collects all yields
    df = pd.DataFrame(data_for_df)        # Then creates DataFrame
    st.write("DataFrame from generator:")
    st.write(df.head(3))
    st.write()

    st.write("5. Demonstrating generator exhaustion:")
    gen = demo_generator()
    # Get first two payments
    payment1 = next(gen)
    payment2 = next(gen)
    st.write(f"Payment 1: {dict(payment1)}")
    st.write(f"Payment 2: {dict(payment2)}")

    # Try to use generator again
    st.write("\nTrying to create new DataFrame from used generator:")
    df_empty = pd.DataFrame(gen)  # Will only get remaining payments
    st.write(f"DataFrame size: {len(df_empty)} rows")  # Will be smaller!

    st.write("\n6. Creating new generator starts fresh:")
    df_full = pd.DataFrame(demo_generator())  # Creates new generator
    st.write(f"New DataFrame size: {len(df_full)} rows")  # Gets all payments

    def demo_generator_with_storage():
        """
        Modified version that stores its own history.
        """
        payment_history = []
        period = 1
        balance = 1000
        
        while balance > 0:
            payment = OrderedDict([
                ('Period', period),
                ('Balance', balance)
            ])
            payment_history.append(payment)
            
            # Yield just the current payment, but we keep the history
            yield payment, payment_history  # Return both current and history
            
            period += 1
            balance -= 100

    st.write("\n7. Generator that maintains its own history:")
    gen_with_history = demo_generator_with_storage()
    for _ in range(3):
        current, history = next(gen_with_history)
        st.write(f"\nCurrent payment: {dict(current)}")
        st.write(f"Number of payments in history: {len(history)}")


with st.expander('Claude example generator'):
    from collections import OrderedDict

    def payment_generator():
        """
        Simple generator to demonstrate how new data is created each cycle.
        """
        loan_balance = 1000
        payment = 400
        period = 1
        
        st.write(f"Generator initialized with starting balance: ${loan_balance}")
        
        while loan_balance > 0:
            # Create new, unique data for this period
            current_payment = min(payment, loan_balance)
            new_balance = loan_balance - current_payment
            
            st.write(f"\nCycle {period}:")
            st.write(f"Creating new OrderedDict for Period {period}")
            st.write(f"Balance at start: ${loan_balance}")
            st.write(f"Payment amount: ${current_payment}")
            st.write(f"New balance: ${new_balance}")
            
            # Yield this period's unique data
            yield OrderedDict([
                ('Period', period),
                ('Starting_Balance', loan_balance),
                ('Payment', current_payment),
                ('Ending_Balance', new_balance)
            ])
            
            # Update for next cycle
            loan_balance = new_balance
            period += 1

    # Demonstrate the cycle-by-cycle creation of data
    st.write("Creating generator...")
    gen = payment_generator()

    st.write("\nGetting first period's data...")
    period_1 = next(gen)
    st.write(f"Period 1 data received: {dict(period_1)}")

    st.write("\nGetting second period's data...")
    period_2 = next(gen)
    st.write(f"Period 2 data received: {dict(period_2)}")

    st.write("\nGetting third period's data...")
    period_3 = next(gen)
    st.write(f"Period 3 data received: {dict(period_3)}")
