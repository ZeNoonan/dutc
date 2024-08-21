import pandas as pd
import numpy as np
import streamlit as st
from itertools import islice, tee
st.set_page_config(layout="wide")

board = [['o', 'o', 'o'],
    ['x', 'x', 'x'],
    ['x', 'o', 'o']]

def rows(board):
    yield from (tuple(x) for x in board)

def row(board):
    yield board

# print(*rows(board))
st.write('rows',*rows(board))
for row in rows(board):
    st.write('row', row)
# st.write(*row(board))

def transpose(board):
    yield from zip(*board)

def transpose_1(board):
    yield zip(*board)




def columns(board):
    yield from transpose(board)

print(*columns(board))
st.write('cols',*columns(board))

def ldiag(board):
    container_tuple=[]
    for i, diag in enumerate(rows(board),start=0):
        container_tuple.append(diag[i])
    yield tuple(container_tuple)

def rdiag(board):
    container_tuple=[]
    for i, diag in enumerate(rows(board),start=1):
        container_tuple.append(diag[-i])
    yield tuple(container_tuple)


st.write('ldiag', * ldiag(board))

def winner(board):
    uniq_row = set(row) - {'.'}
    if len(uniq_row)==1:
        return True, next(iter(uniq_row))
    return False, '.'

def winner_2(board):
    uniq_row = set(row) - {'.'}
    if len(uniq_row)==1:
        yield True, next(iter(uniq_row))
    yield False, '.'

st.write('board unpacked', *rows(board))
st.write('this produces incorrect result',winner(rows(board))) # interesting that this produces wrong result
# st.write(winner(*rows(board))) # this unpacks the 3 rows into the function which doesn't work
for row in rows(board):
    st.write('row winner?', winner(row))
# how do i write the above using yield from

def row_winner_function(x):
    yield from winner_2(x)

# won,victor = row_winner_function(rows(board))
# st.write(won,victor)

st.write(*row_winner_function(rows(board)))
for x in row_winner_function(rows(board)):
    st.write(x)

st.write('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

board = [['10', '11', '12'],
    ['13', '14', '15'],
    ['16', '17', '18']]

board = [
    ['1', '2', '3', '4'],
    ['5', '6', '7', '8'],
    ['9', '10', '11', '12'],
    ['13', '14', '15', '16']
]

def traverse(board):
    for trav in [rows, columns, rdiag, ldiag]:
        for i, group in enumerate(trav(board)):
            yield trav, i, group

# def traverse_other(board):
#     for trav in [rdiag, ldiag]:
#         for i, group in enumerate(trav(board)):
#             st.write('group', group)
#             yield trav, i, group

with st.expander('Nwise function workings'):
    st.dataframe(board)
    st.write(*(f'{trav.__name__: <8} {i}: {group}'for trav, i, group in traverse(board)))
    
    def nwise(iterable, n=2):
        return zip(
            *(islice(g, i, None) for i, g in enumerate(tee(iterable, n))))
    
    data = [11, 12, 13, 14, 15]
    # for group in nwise_chatgpt(data, 3):
    #     st.write('chatgpt implementation,group)

    data_test=iter(data)
    for i, g in enumerate(tee(iter(data), 3)):
        st.write('i',i,'g',*islice(g,i,None))
    st.write('I see how it works now so take the above output and then do a zip on it')
    st.write('zip takes the first element in each and then discards anything that doesnt make up the tuple of 3')
    st.write('oh no just read that zip only works for the smallest list passed which now works')

    st.write('so now need to put into words how it works')
    st.write('nwise produces a certain number of groups of digits which are consecutively ordered and it slides the window forward one amount at a time')
    st.write('so its like a moving window where you get to pick how many digits are in your list you want')
    st.write('say you have a list of 10 digits, and you want an 8 digit list, then your group size is 8, then it starts counting from 1-8 which is one group, then 2-9 another group and 3-10 last group')
    st.write('the tee iterable creates lets say 3 instances of the list it also corresponds to how many digits do you want in your list')


    data_iter=iter(data)
    for group in nwise(data_iter, 3):
        st.write('nwise j powell',group)



    # got the below from chat gpt really helpful as was trying to do it myself
    def nwise_print(iterable, n=2):
        # Create independent iterators
        iterators = tee(iterable, n)
        st.write("Iterators from tee:", iterators)

        # Enumerate the iterators
        indexed_iterators = list(enumerate(iterators))
        st.write("Enumerated iterators:", indexed_iterators)

        # Use islice to create new iterators starting from index i
        sliced_iterators = [list(islice(g, i, None)) for i, g in indexed_iterators]
        st.write("Sliced iterators:", sliced_iterators)

        # Zip the sliced iterators
        result = list(zip(*sliced_iterators))
        st.write("Result after zip:", result)

        return result

    data = [11, 12, 13, 14, 15]
    data_1 = [11, 12, 13, 14, 15]
    # Call the nwise function
    result = nwise_print(data, 3)
    result_1 = nwise_print(data_1, 2)

    # Print the final result
    st.write("Final result:", result)
    st.write("Final result:", result_1)




with st.expander('length of group and continue'):
    with st.echo():
        for trav, i, group in traverse(board):
            st.write('length of group',len(group))
            st.write('i',i)
            if len(group) < 3: continue
            # for offset, line in enumerate(nwise(group, n=4)):
            #     st.write(offset,line)
            # for line in nwise(group, n=3):
            #     st.write('this is line',line)

st.write('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

with st.echo():
    # pass
    # test=range(6)
    # for x in islice(test,2,4):
    #     st.write(x) # takes no keyword args, so data then start then stop but defaults to stop if only data plus one arg
    
    st.write(*(islice(g, i, None) for i, g in enumerate(tee(range(6), 2))))
    for i, g in enumerate(range(6)):
        st.write((g,i))

def nwise_chatgpt(iterable, n):
    """Generate overlapping groups of n elements from the given iterable."""
    it = iter(iterable)
    result = tuple(islice(it, n))
    while len(result) == n:
        yield result
        result = result[1:] + (next(it),)

# Example usage:

# with st.echo():
#     st.write(*(f'{trav.__name__: <8} {i}: {group}'for trav, i, group in traverse_other(board)))


def find_winners(board):
    for trav, i, group in traverse(board):
        won, pl = winner(group)
        if won:
            yield trav, i, pl

# st.write(
#     *(f'{pl!r} won on {trav.__name__: <8} {i}'
#     for trav, i, pl in find_winners(board))
# )

st.write('yyyyyyyyyyyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

# with st.echo():
#     st.write(*(f'{trav,i,group}' for trav, i, group in traverse(board)))
# st.write(*(trav,i,group for trav, i, group in traverse(board)))

# st.write(
#     *(f'{trav.__name__: <8} {i}: {group}'
#     for trav, i, group in traverse(board)),
#     sep='\n',
# )

board = [
    ['x', 'o', 'o', 'x'],
    ['o', 'o', 'o', 'o'],
    ['x', 'x', 'x', 'o'],
    ['x', 'o', 'o', 'x']
]                

st.write(
    *(f'{trav.__name__: <8} {i}: {group}'
    for trav, i, group in traverse(board)),
    sep='\n',
)

board = [
    ['1', '2', '3', '4'],
    ['5', '6', '7', '8'],
    ['9', '10', '11', '12'],
    ['13', '14', '15', '16']
]

with st.echo():
    st.write('length of board',len(board))
    offset_workings=[1,2,3]
    i_workings=[]
    # for i in range(1):
    #     st.write('this is i', i)
    st.write('Board 0,-1:',board[0][-1])


    for offset in range(1, len(board)):
        # st.write(board[i][i - offset])
        st.write(board[i][i - offset] for i in range(offset))
        for i in range(offset):
            # st.write('offset equals length of board:',offset)
            # st.write('i:',i)
            # st.write('i-offset:',i-offset)
            st.write(board[i][i - offset])

    for offset in reversed(range(1, len(board))):
        st.write('reveesed:', offset)
            
    def major_diags_print(board):
        # above major diagonals
        for offset in range(1, len(board)):
            yield tuple(board[i][i - offset] for i in range(offset))

    st.write(*{major_diags_print(board)})

def major_diags(board):
    # above major diagonals
    for offset in range(1, len(board)):
        yield tuple(
            board[i][i - offset] for i in range(offset)
        )
        
    # major diagonal
    yield tuple(row[i] for i, row in enumerate(rows(board)))
    
    # below diagonals
    for offset in reversed(range(1, len(board))):
        yield tuple(
            board[i - offset][i] for i in range(offset)
        )

board = [
    ['x', 'x', 'o', 'x'],
    ['o', 'o', 'o', 'o'],
    ['x', 'x', 'x', 'o'],
    ['x', 'o', 'o', 'x']
]

major = {
    ('x', ),
    ('o', 'o',),
    ('x', 'o', 'o',),
    ('x', 'o', 'x', 'x',),
    ('o', 'x', 'o',),
    ('x', 'o',),
    ('x',),
}

assert major == {*major_diags(board)}
st.write({*major_diags(board)})
st.write(*major_diags(board))

# for offset in range(1, len(board)):
#             st.write(board[i][i - offset] for i in range(offset))


def major_diags(board):
    # above major diagonals
    for offset in range(1, len(board)):
        yield tuple(
            board[i][i - offset] for i in range(offset)
        )
        

board = [
    ['x', 'x', 'o', 'x'],
    ['o', 'o', 'o', 'o'],
    ['x', 'x', 'x', 'o'],
    ['x', 'o', 'o', 'x']
]

board = [
    ['1', '2', '3', '4'],
    ['5', '6', '7', '8'],
    ['9', '10', '11', '12'],
    ['13', '14', '15', '16']
]


# assert major == {*major_diags(board)}
st.write({*major_diags(board)})
st.write(*major_diags(board))

st.write('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

with st.expander('Checking out the winner function'):

    def nwise(iterable, n=2):
        return zip(
            *(islice(g, i, None) for i, g in enumerate(tee(iterable, n)))
        )

    def winner(line, ignore={'.'}):
        uniq_line = set(line)
        if len(uniq_line) == 1 and uniq_line != ignore:
            return True, next(iter(uniq_line))
        return False, None


    def find_winners(board, size=3):
        for trav, i, group in traverse(board):
            if len(group) < size: continue
            st.write(f"\nChecking {trav.__name__} {i}:")
            for offset, line in enumerate(nwise(group, n=size)):
                st.write(f"  Offset {offset}: Checking line {line}")
                won, pl = winner(line)
                if won:
                    st.write(f"  Winner found! Player {pl} won on {trav.__name__} {i}+{offset}")
                    yield trav, i, offset, pl
        
    board = [
        ['x', 'o', 'x', 'x'],
        ['o', '.', 'o', 'o'],
        ['o', 'x', 'x', 'x'],
        ['o', 'x', 'o', 'o']
    ]        

    for size in range(3, len(board)+1):
        st.write(
            
            f'{size} in a line constitutes a win!',
            *(f'{pl!r} won on {trav.__name__: <14} {i}+{offset}'
            for trav, i, offset, pl in find_winners(board, size=size)),
            
        )