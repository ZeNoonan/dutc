import pandas as pd
import numpy as np
import streamlit as st

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

board = [
    ['o', 'o', 'o'],
    ['x', 'x', 'x'],
    ['x', 'o', 'o']
]

print(
    'Rows', *rows(board),
    'Columns', *columns(board),
    'left diagonal', [*ldiag(board)],
    'right diagonal', [*rdiag(board)],
    sep='\n',
)

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

game_winner(board) # where did the win occur? a row, column, diagonal?

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

board = [
    ['o', 'o', 'o'],
    ['x', 'x', 'x'],
    ['x', 'o', 'o']
]

print(
    'Rows', *rows(board),
    'Columns', *columns(board),
    'left diagonal', *ldiag(board),
    'right diagonal', *rdiag(board),
    sep='\n',
)

def traverse(board):
    for trav in [rows, columns, rdiag, ldiag]:
        for i, group in enumerate(trav(board)):
            yield trav, i, group


board = [
    ['o', 'o', 'o'],
    ['x', 'x', 'x'],
    ['x', 'o', 'o']
]                

print(
    *(f'{trav.__name__: <8} {i}: {group}'
    for trav, i, group in traverse(board)),
    sep='\n',
)

def find_winners(board):
    for trav, i, group in traverse(board):
        won, pl = winner(group)
        if won:
            yield trav, i, pl
            
print(
    *(f'{pl!r} won on {trav.__name__: <8} {i}'
    for trav, i, pl in find_winners(board)),
    sep='\n'
)

# https://www.dontusethiscode.com/blog/2023-11-08_tictactoe.html
###########################################################################################

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

print(
    *(f'{trav.__name__: <8} {i}: {group}'
    for trav, i, group in traverse(board)),
    sep='\n',
)

def test_row_win():
    boards = [
        [
            ['x', 'x', 'x'],
            ['x', 'o', 'o'],
            ['o', 'x', 'x'],
        ],
        [
            ['x', 'o', 'o'],
            ['x', 'x', 'x'],
            ['o', 'x', 'x'],
        ]
    ]

    for b in boards:
        won, victor = game_winner(b)
        assert won
        assert victor == 'x'

def test_col_win():
    boards = [
        [
            ['o', 'o', 'x'],
            ['o', 'x', 'o'],
            ['o', 'o', 'x'],
        ],
        [
            ['x', 'o', 'o'],
            ['o', 'o', 'x'],
            ['o', 'o', 'x'],
        ]
    ]
    for b in boards:
        won, victor = game_winner(b)
        assert won
        assert victor == 'o'

def test_rows():
    board = [
        ['x', 'x', 'x'],
        ['x', 'o', 'o'],
        ['o', 'x', 'x'],
    ]

    assert [*rows(board)] == board

def test_columns():
    board = [
        ['x', 'x', 'x'],
        ['x', 'o', 'o'],
        ['o', 'x', 'x'],
    ]

    assert [*columns(board)] == [
        ['x', 'x', 'o'],
        ['x', 'o', 'x'],
        ['x', 'o', 'x'],
    ]



test_row_win()

board = [
    ['x', 'o', 'o', 'x'],
    ['o', 'o', 'o', 'o'],
    ['x', 'x', 'x', 'o'],
    ['x', 'o', 'o', 'x']
]                

print(
    *(f'{trav.__name__: <8} {i}: {group}'
    for trav, i, group in traverse(board)),
    sep='\n',
)

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

def flip_lr(board):
    """reverses each row of the board
    """
    yield from (r[::-1] for r in rows(board))

def minor_diags(board):
    yield from major_diags([*flip_lr(board)])

board = [
    ['x', 'x', 'o', 'x'],
    ['o', 'o', 'o', 'o'],
    ['x', 'x', 'x', 'o'],
    ['x', 'o', 'o', 'x']
]
    
minor = {
    ('x',),
    ('x', 'o',),
    ('o', 'o', 'x',),
    ('x', 'o', 'x', 'x',),
    ('o', 'x', 'o',),
    ('o', 'o',),
    ('x',),
}

assert major == {*major_diags(board)}
assert minor == {*minor_diags(board)} # not working

def transpose(board):
    yield from zip(*board)

def rows(board):
    yield from (tuple(row) for row in board)

def columns(board):
    yield from transpose(board)

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

# def minor_diags(board):
#     yield from major_diags([*transpose(board)])
            
def traverse(board):
    for trav in [rows, columns, major_diags, minor_diags]:
        for i, group in enumerate(trav(board)):
            yield trav, i, group

board = [
    ['x', 'x', '.', 'x'],
    ['o', '.', 'x', 'o'],
    ['o', 'x', 'x', 'x'],
    ['o', 'o', 'o', 'x']
]            

print(
    *(f'{trav.__name__: <8} {i}: {group}'
    for trav, i, group in traverse(board)),
    sep='\n',
)

from itertools import islice, tee

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
        for offset, line in enumerate(nwise(group, n=size)):
            won, pl = winner(line)
            if won:
                yield trav, i, offset, pl
    
board = [
    ['x', 'x', 'x', 'x'],
    ['o', '.', 'x', 'x'],
    ['o', 'x', 'x', 'x'],
    ['o', 'o', 'o', 'x']
]        

for size in range(3, len(board)+1):
    print(
        '\N{box drawings light horizontal}'*40,
        f'{size} in a line constitutes a win!',
        *(f'{pl!r} won on {trav.__name__: <14} {i}+{offset}'
        for trav, i, offset, pl in find_winners(board, size=size)),
        sep='\n',
    )


