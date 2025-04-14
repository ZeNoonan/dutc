When would I actually use eval or exec… and should I feel as guilty when I do it?
print("Let's take a look!")
In Python, the builtin eval and exec functions allow us to execute code encoded as an str. eval allows us to evaluate a single expression and returns its result; exec allows us to execute a suite of statements but does not return anything.

from textwrap import dedent

code = '1 + 1'
print(f'{eval(code) = }')

code = dedent('''
    x = 1
    y = 1
    x + y
''').strip()
print(f'{exec(code) = }')
With both exec and eval, you can pass in a namespace; with exec, you can capture results by capturing name binding in this namespace

from textwrap import dedent

code = '1 + 1 + z'
print(f'{eval(code, globals(), ns := {"z": 123}) = }')
print(f'{ns = }')

code = dedent('''
    x = 1
    y = 1
    w = x + y + z
''').strip()
print(f'{exec(code, globals(), ns := {"z": 123}) = }')
print(f'{ns = }')
Obviously, eval('1 + 1') is inferior to evaluating 1 + 1. We don’t get syntax highlighting. We don’t get any static mechanisms provided by the interpreter (such as constant folding.)

However, by encoding the executed or evaluated code as a string, that means we can use string manipulation to create code snippets. Obviously, in most cases, this is inferior to other programmatic or meta-programmatic techniques.

x, y, z = 123, 456, 789

var0, var1 = 'x', 'y'
code = f'{var0} + {var1}'
res = eval(code, globals(), locals())
print(f'{res = }')

if ...:
    res = x + y
    print(f'{res = }')

var0, var1 = 'x', 'y'
res = globals()[var0] + globals()[var1]
print(f'{res = }')
But there are also clearly metaprogramming situations where string manipulation may be superior.

from dataclasses import dataclass
from datetime import datetime
from typing import Any

@dataclass
class Propose:
    ident : int
    timestamp : datetime
    payload : Any

@dataclass
class Accept:
    ident : int
    timestamp : datetime

@dataclass
class Reject:
    ident : int
    timestamp : datetime

@dataclass
class Commit:
    ident : int
    timestamp : datetime
    payload : Any

print(
    f'{Propose(..., ..., ...) = }',
    f'{Accept(..., ...)       = }',
    f'{Reject(..., ...)       = }',
    f'{Commit(..., ..., ...)  = }',
    sep='\n',
)
from csv import reader
from textwrap import dedent
from dataclasses import dataclass

message_definitions = dedent('''
    name,*fields
    Propose,ident,timestamp,payload
    Acc ept,ident,timestmap
    Reject,ident,timestmap
    Commit,ident,timestamp,payload
''').strip()
messages = {}
for lineno, (name, *fields) in enumerate(reader(message_definitions.splitlines()), start=1):
    if lineno == 1: continue
    messages[name] = name, fields

class MessageBase: pass
for name, fields in messages.values():
    globals()[name] = dataclass(type(name, (MessageBase,), {
        '__annotations__': dict.fromkeys(fields)
    }))

print(
    globals(),
    f'{Propose(..., ..., ...) = }',
    # f'{Accept(..., ...)       = }',
    f'{Reject(..., ...)       = }',
    f'{Commit(..., ..., ...)  = }',
    sep='\n',
)
from csv import reader
from textwrap import dedent, indent
from dataclasses import dataclass

message_definitions = dedent('''
    name,*fields
    Propose,ident,timestamp,payload
    Accept,ident,timestmap
    Reject,ident,timestmap
    Commit,ident,timestamp,payload
''').strip()
messages = {}
for lineno, (name, *fields) in enumerate(reader(message_definitions.splitlines()), start=1):
    if lineno == 1: continue
    messages[name] = name, fields

class MessageBase: pass
for name, fields in messages.values():
    code = dedent(f'''
        @dataclass
        class {name}(MessageBase):
        
    ''').strip().format(fields=indent('\n'.join(f"{f} : ..." for f in fields), ' ' * 4))
    print(code)
    exec(code, globals(), locals())

print(
    f'{Propose(..., ..., ...) = }',
    f'{Accept(..., ...)       = }',
    f'{Reject(..., ...)       = }',
    f'{Commit(..., ..., ...)  = }',
    sep='\n',
)
There is nothing inherently wrong about eval or exec (in most execution environments.)

from tempfile import TemporaryDirectory
from sys import path
from pathlib import Path
from textwrap import dedent

with TemporaryDirectory() as d:
    d = Path(d)
    code = dedent('''
        class T: pass
    ''').strip()
    with open(d / 'module.py', mode='wt') as f:
        print(code, file=f)
    path.insert(0, f'{d!s}')
    import module
    del path[0]

print(f'{module.T = }')
We can think of all code creation mechanisms as lying on a spectrum:

from tempfile import TemporaryDirectory
from inspect import getsource
from collections import namedtuple
from textwrap import dedent
from sys import path
from pathlib import Path

class T0: pass

T1 = namedtuple('T1', '')
class T1(namedtuple('T1', '')): pass

...

T2 = type('T2', (tuple,), {'__call__': lambda _: ...})

...

exec(dedent('''
    class T3:
        pass
'''), globals(), locals())

with TemporaryDirectory() as d:
    d = Path(d)
    code = dedent('''
        class T4:
            pass
    ''').strip()
    with open(d / 'module.py', mode='wt') as f:
        print(code, file=f)
    path.insert(0, f'{d!s}')
    from module import *
    del path[0]

    print(
        f'{T0 = }', # getsource(T0),
        f'{T1 = }', # getsource(T1),
        f'{T2 = }', # getsource(T2),
        f'{T3 = }', # getsource(T3),
        f'{T4 = }', getsource(T4),
        sep='\n',
    )