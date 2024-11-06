import streamlit as st
from collections import UserDict
from inspect import getouterframes, currentframe
from itertools import chain
from pathlib import Path
from io import StringIO
from textwrap import indent

st.set_page_config(layout="wide")

# https://www.dontusethiscode.com/blog/2024-10-23_global_state_refactor.html

def logframe(frame, buffer=None):
    if buffer is None:
        buffer = StringIO()
        
    template = '{f.filename}:{f.function}:{f.lineno} → {code_context}'.format
    frame_iter = reversed(getouterframes(frame))
    
    # Skip past async code dispatch lines (specific to Jupyter)
    for f in frame_iter:
        if f.filename.startswith('/tmp'):
            break
    
    # Log information to buffer
    for f in chain([f], frame_iter):
        buffer.write(template(f=f, code_context=''.join(f.code_context).strip()))
        buffer.write('\n')
    return buffer.getvalue()

class D(UserDict):
    def __init__(self, *args, **kwargs):
        self.log = False
        super().__init__(*args, **kwargs)
        self.log = True

    def __getitem__(self, key):
        if self.log:
            print(
                f'__getitem__ {key!r}',
                indent(logframe(currentframe().f_back), prefix=' '*2),
                sep='\n'
            )
        return self.__dict__['data'][key]
    
    def __setitem__(self, key, value):
        if self.log:
            print(
                f'__setitem__ {key!r}',
                indent(logframe(currentframe().f_back), prefix=' '*2),
                sep='\n'
            )
        self.__dict__['data'][key] = value

variables = {
    'x': 1,
    'y': ['hello', 'world'],
    'z': 12,
    'mu': 42,
    'category': 'something',
}


variables = D(variables)

def f():
    return ' '.join(variables['y'])

def g():
    if variables['category'] == 'something':
        variables['mu'] /= 3
    return variables['mu']

def h():
    z = variables['z']
    def _h():
        return variables['x'] + z
    return _h()


st.write(
    f'{f() = }',
    f'{g() = }',
    f'{h() = }',
    f'{variables.pop("mu") = }',
    sep='\n'
)