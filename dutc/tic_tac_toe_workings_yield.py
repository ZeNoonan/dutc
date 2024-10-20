import pandas as pd
import numpy as np
import streamlit as st
st.set_page_config(layout="wide")

def transpose(board):
    yield from zip(*board)

def winner(row):
    uniq_row = set(row) - {'.'}
    if len(uniq_row) == 1:
        return True, next(iter(uniq_row))
    return False, '.'

def rows(board):
    yield from board

def columns(board):
    yield from transpose(board)

def ldiag(board):
    for i, row in enumerate(rows(board)):
        yield row[i]

def rdiag(board):
    for i, row in enumerate(rows(board), start=1):
        yield row[-i]

def game_winner(board):
    # check if there is a win in the row
    for i, row in enumerate(rows(board)):
        won, victor = winner(row)
        if won:
            return won, victor

    # check if there is a win in the column
    for i, col in enumerate(columns(board)):
        won, victor = winner(col)
        if won:
            return won, victor

    # check if there is a win in the left diagonal
    won, victor = winner(ldiag(board))
    if won:
        return won, victor

    # check if there is a win in the right diagonal
    won, victor = winner(rdiag(board))
    if won:
        return won, victor
    return False, '.'

# game_winner(board) # where did the win occur? a row, column, diagonal?


board = [['o', 'o', 'o'],
    ['x', 'x', 'x'],
    ['x', 'o', 'o']]

print(
    'Rows', *rows(board),
    'Columns', *columns(board),
    'left diagonal', [*ldiag(board)],
    'right diagonal', [*rdiag(board)],
    sep='\n',
)
with st.expander('output of initial code'):
    st.write(
        'Rows', *rows(board),
        'Columns', *columns(board),
        'left diagonal', [*ldiag(board)],
        'right diagonal', [*rdiag(board)],
        sep='\n',
    )

with st.expander('Work on rows'):
    pass

with st.expander('Work on asteriks unpacking'):
    st.write('this is just pringint out board', board)
    st.write('this is using the sorted function instead of generator with asteriks',*sorted(board))
    st.write('rows yield from no asteriks', rows(board))
    

    def rows_yield(board):
        yield board

    # st.write(*sorted(rows_yield))
    st.write('rows yield with no asteriks', rows_yield(board))
    st.write('rows function with yield (no yield from) with asteriks', *rows_yield(board))
    st.write('rows "yield from" function with asteriks', *rows(board))

with st.expander('Below is work on the cols zip function'):
    st.write('now need to look at the zip and asteriks in function')
    with st.echo():
        for x in zip(*board):
            st.write('x in zip board',x)
        st.write('so the yield from chains up the output instead of writing a for loop')
    st.write('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    def transpose(board):
        yield from zip(*board)

    def columns(board):
        yield from transpose(board)

    st.write('using yield from in the zip transpose function', *transpose(board))
    st.write('using yield from in the full columns function', *columns(board))
    st.write('interesting that there is no difference between the above keep an eye on this')

with st.expander('work on diagonal'):
        board = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
        for i, x in enumerate(rows(board),start=1):
            st.write( 'this is x-1',x[-i])

with st.expander('work on game winner'):
    
    board = [
    ['x', 'o', '.'],
    ['x', 'o', 'x'],
    ['x', 'x', 'o']]

    def winner_1(row):
        st.write('set',set(row))
        # st.write("{'.'}",{'.'})
        uniq_row = set(row) - {'.'}
        st.write('unique row', uniq_row)
        if len(uniq_row) == 1:
            return True, next(iter(uniq_row))
        return False, '.'
    
    def game_winner_1(board):
    # check if there is a win in the row
        for i, row in enumerate(rows(board)):
            won, victor = winner_1(row)
            if won:
                return won, victor
        return False, '.'
            
    st.write(game_winner_1(board))
    st.write('so chat gpt provided some value here, couldnt wrap my head around the set minus . but it turns out that . is supposed to represent \
             an empty cell so i guess if you filled it in with a dot it would just take it out')
    
    st.write('in terms of understanding the next(iter(uniq_row)) chat gpt was really good copying the answer below')
    st.write("next(iter(uniq_row)): This retrieves the next element from the iterator. Since we know that uniq_row has only one element\
            (as per the previous check len(uniq_row) == 1), this effectively gets that single element from the set.\
            So, in the context of the code, if there is a winner in the row (i.e., len(uniq_row) == 1), the next(iter(uniq_row)) part\
            is used to obtain the winning element from the set uniq_row. This element is then returned along with True to\
            indicate that there is a winner in the row. In summary, next(iter(uniq_row)) is a concise way to retrieve the single element from a set,\
            and it's used here to get the winning element when there is a winner in the row.")
    

with st.expander('2nd Pass'):
    def transpose(board):
        yield from zip(*board)

    def winner(row):
        uniq_row = set(row) - {'.'}
        if len(uniq_row) == 1:
            return True, next(iter(uniq_row))
        return False, '.'

    def rows(board):
        yield from (tuple(row) for row in board)

    def columns(board):
        yield from transpose(board)

    def ldiag(board):
        diag = []
        for i, row in enumerate(rows(board), start=0):
            diag.append(row[i])
        yield tuple(diag)

    def rdiag(board):
        diag = []
        for i, row in enumerate(rows(board), start=1):
            diag.append(row[-i])
        yield tuple(diag)

    def traverse(board):
        for trav in [rows, columns, rdiag, ldiag]:
            for i, group in enumerate(trav(board)):
                yield trav, i, group


    board = [
        ['o', 'o', 'o'],
        ['x', 'x', 'x'],
        ['x', 'o', 'o']
    ]                

    st.write(
        *(f'{trav.__name__: <8} {i}: {group}'
        for trav, i, group in traverse(board)),
        sep='\n',
    )

    def find_winners(board):
        for trav, i, group in traverse(board):
            won, pl = winner(group)
            if won:
                yield trav, i, pl
            
    st.write(
        *(f'{pl!r} won on {trav.__name__: <8} {i}'
        for trav, i, pl in find_winners(board)),
        sep='\n'
    )