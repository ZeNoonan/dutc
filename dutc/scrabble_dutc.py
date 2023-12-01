import pandas as pd
import numpy as np
import streamlit as st


# <!-- Requirements
# Simulate a single turn of the game of Scrabble.
# Blank tiles will not be used in this version of the game.
# Derived

# Draw N tiles from a population of tiles.
# Using the above N tiles, we need figure out what words can make.
# Using associated point values for each letter, determine the highest scoring word we can make. -->

words = ['hello', 'world', 'test', 'python', 'think']
tiles = [*'pythnoik']

for w in words:
    print(f'{sorted(w) == sorted(tiles) = }')
def letter_counts(text):
    tile_counts = {}
    for letter in text:
        tile_counts[letter] = tile_counts.get(letter, 0) + 1
    return tile_counts

def find_candidates(letters, words):
    tile_counts = letter_counts(letters)
    for w in words:
        word_counts = letter_counts(w)

        if word_counts.keys() <= tile_counts.keys():
            if all(tile_counts[l] >= c for l, c in word_counts.items()):
                yield w

words = ['hello', 'world', 'test', 'python', 'think']
tiles = [*'pythnoik']
# tiles = [*'hello']
print([*find_candidates(tiles, words)], sep='\n')

text = '''
1  - A, E, I, O, U, L, N, S, T, R
2  - D, G
3  - B, C, M, P
4  - F, H, V, W, Y
5  - K
8  - J, X
10 - Q, Z
'''.strip()

points = {}
for line in text.splitlines():
    value, _, letters = line.partition('-')
    for l in letters.strip().split(', '):
        points[l] = int(value)
print(dict(sorted(points.items())))

POINTS = {
    'A': 1, 'E': 1, 'I': 1, 'O': 1, 'U': 1,
    'L': 1, 'N': 1, 'S': 1, 'T': 1, 'R': 1,
    'D': 2, 'G': 2, 'B': 3, 'C': 3, 'M': 3,
    'P': 3, 'F': 4, 'H': 4, 'V': 4, 'W': 4,
    'Y': 4, 'K': 5, 'J': 8, 'X': 8, 'Q': 10, 'Z': 10
}
POINTS = {k.casefold(): v for k, v in POINTS.items()}

def letter_counts(text):
    tile_counts = {}
    for letter in text:
        tile_counts[letter] = tile_counts.get(letter, 0) + 1
    return tile_counts

def find_candidates(letters, words):
    tile_counts = letter_counts(letters)
    for w in words:
        word_counts = letter_counts(w)

        if word_counts.keys() <= tile_counts.keys():
            if all(tile_counts[l] >= c for l, c in word_counts.items()):
                yield w

def score(word, points):
    cur_score = 0
    for letter in word:
        cur_score += points[letter]
    return cur_score


words = ['hello', 'world', 'test', 'python', 'think']
tiles = [*'pythnoik']
# tiles = [*'hello']
candidates = [*find_candidates(tiles, words)]
print(
    max(candidates, key=lambda s: score(s, POINTS))
)

# How to draw tiles from a population?

text = '''
A-9, B-2, C-2, D-4, E-12, F-2, G-3, H-2, I-9, J-1, K-1, L-4, M-2, N-6, O-8, P-2, Q-1, R-6, S-4, T-6, U-4, V-2, W-2, X-1, Y-2, Z-1
'''.strip()

population = {}
for entry in text.split(', '):
    letter, value = entry.split('-')
    population[letter.casefold()] = int(value)
print(population)

# file: scabble.py

from collections import Counter
from functools import wraps
from random import Random

POPULATION = {
    'a': 9, 'b': 2, 'c': 2, 'd': 4, 'e': 12, 'f': 2, 'g': 3,
    'h': 2, 'i': 9, 'j': 1, 'k': 1, 'l': 4, 'm': 2, 'n': 6,
    'o': 8, 'p': 2, 'q': 1, 'r': 6, 's': 4, 't': 6, 'u': 4,
    'v': 2, 'w': 2, 'x': 1, 'y': 2, 'z': 1
}
POINTS = {
    'A': 1, 'E': 1, 'I': 1, 'O': 1, 'U': 1,
    'L': 1, 'N': 1, 'S': 1, 'T': 1, 'R': 1,
    'D': 2, 'G': 2, 'B': 3, 'C': 3, 'M': 3,
    'P': 3, 'F': 4, 'H': 4, 'V': 4, 'W': 4,
    'Y': 4, 'K': 5, 'J': 8, 'X': 8, 'Q': 10, 'Z': 10
}
POINTS = {k.casefold(): v for k, v in POINTS.items()}

def letter_counts(text):
    tile_counts = {}
    for letter in text:
        tile_counts[letter] = tile_counts.get(letter, 0) + 1
    return tile_counts

def find_candidates(letters, words):
    tile_counts = letter_counts(letters)
    for w in words:
        word_counts = letter_counts(w)

        if word_counts.keys() <= tile_counts.keys():
            if all(tile_counts[l] >= c for l, c in word_counts.items()):
                yield w

def score(word, points):
    cur_score = 0
    for letter in word:
        cur_score += points[letter.casefold()]
    return cur_score


def pump(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        gen = f(*args, **kwargs)
        next(gen)
        return gen
    return wrapper

@pump
def draw_tiles(population, *, rnd=None):
    if rnd is None: rnd = Random()

    population = Counter(population)
    k = yield
    while population.total():
        k = min(k, population.total())
        res = rnd.sample([*population.keys()], k=k, counts=population.values())
        population -= Counter(res)
        k = yield res

words = []
with open('C:/Users/Darragh/Documents/Python/dutc/words.txt') as f:
    for word in f:
        word = word.strip()
        if word.isalpha() and len(word) >= 3 and word.islower():
            words.append(word)

sampler = draw_tiles(POPULATION, rnd=Random(0))
hand = sampler.send(8)

print(
    hand,
    # *find_candidates(hand, words),
    max(find_candidates(hand, words), key=lambda s: score(s, POINTS)),
    sep='\n'
)

# tests

# from scabble import score, POINTS

from hypothesis import given
from hypothesis.strategies import text

from pytest import raises

@given(
    word=text(min_size=1)
)
def test_score(word):
    if word.isalpha():
        total_score = score(word, POINTS)
    else:
        with raises(KeyError):
            total_score = score(word, POINTS)
