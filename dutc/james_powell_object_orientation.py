What do I need to know about object orientation in Python? What is an example convention? What is an example rule? … and does this make things more understandable?
print("Let's take a look!")
The Python object model is very “mechanical,” and our understanding of many of the protocol methods may be little more than a reflection of this mechanical understanding.

For example, when instances are created, we call __new__ prior to instance creation and __init__ afterwards. This immediately gives us an indication for when we may want to implement __new__ vs __init__.

class TMeta(type):
    def __call__(cls):
        obj = cls.__new__(cls) # ...
        cls.__init__(obj)      # ...
        return obj

class T(metaclass=TMeta):
    def __new__(cls):
        return super().__new__(cls)
    def __init__(self):
        pass

obj = T()
print(f'{obj = }')
For other protocol methods, we may need to dig a bit deeper to discover the underlying meaning. For example, __repr__ is the human readable representation of an object, and __str__ is documented as the “informal” printable representation.

However, when we consider that __str__ is triggered by str(…), we can derive an alternate meaning for __str__: it is the data represented in the form of an str.

from dataclasses import dataclass
from enum import Enum

class Op(Enum):
    Eq = '='
    Lt = '<'
    Gt = '>'
    ...
    def __str__(self):
        return self.value

@dataclass
class Where:
    column : str
    op : Op
    value : ...
    def __str__(self):
        return f'{self.column} {self.op} {self.value}'

@dataclass
class Select:
    exprs : list[str]
    table : str
    where : Where | None

    def __str__(self):
        where = f' {self.where}' if self.where else ''
        return f'select {", ".join(self.exprs)} from {self.table}{where}'

stmt = Select(
    ['name', 'value'],
    'data',
    Where('value', Op.Gt, 0),
)

from pathlib import Path

d = Path('/tmp')

print(
    # f'{str(d)       = }',
    # f'{str(123.456) = }',
    # f'{repr(stmt) = }',
    # f'{str(stmt)  = }',
    sep='\n',
)
Some protocol methods are misleading. For example, it may appear that __hash__ means “a pigeonholed identifier,” but its meaning is far narrower.

from dataclasses import dataclass

@dataclass
class T:
    value : ...
    def __hash__(self):
        return hash(self.value)

obj = T((1, 2, 3))
print(
    f'hash(obj) = {hash(obj)}',
)
For some protocol methods, we need to pay close attention to the implementation rules. For example, __len__ means the “size” of an object, where that concept of size must be the “integer, non-negative” size.

class T:
    def __len__(self):
        # return -2
        # return 2.5
        return 2

obj = T()
print(f'{len(obj) = }')
Sometimes, there is disagreement about the implicit rules of implementation.

python -m pip install numpy
class T:
    def __bool__(self):
        raise ValueError(...)
        # return ...

bool(T())
from enum import Enum
from numpy import array

class Directions(Enum):
    North = array([+1,  0])
    South = array([-1,  0])
    East  = array([ 0, +1])
    West  = array([ 0, -1])

print(
    array([0, 0]) + Directions.North * 2
)
In fact, even PEP-8 makes this mistake:

xs = [1, 2, 3]

if len(xs) > 0: pass
if not len(xs): pass
if xs: pass # preferred
from numpy import array
xs = array([1, 2, 3])

if len(xs) > 0: pass
if not len(xs): pass
if xs.size > 0: pass
# if xs: pass # ValueError
As we can see __bool__ should return True or False but, in the case of a numpy.ndarray or pandas.Series, instead raises a ValueError.

python -m pip install numpy pandas
from numpy import array
from pandas import Series

xs = array([1, 2, 3])
s = Series([1, 2, 3])

print(
    # f'{bool(xs) = }',
    # f'{bool(s)  = }',
    sep='\n',
)
Of course, in the PEP-8 example, this isn’t altogether that meaningful of a problem.

from numpy import array

xs = [1, 2, 3]
xs.append(4)
xs.clear()

if not xs:
    pass
for x in xs:
    pass

xs = array([1, 2, 3])
xs = xs[xs > 10]

if not xs:
    pass
Note that the entire reason we are choosing to interact with the Python “vocabulary” is to be able to write code that is obvious to the reader.

...
...
...
...
# try:
v = obj[k]
# except LookupError:
#     pass
...
...
...
...
This means that when we implement data model methods, we should implement them only where their meaning is unambiguous. This suggests that the implementation of these methods should be to support a singular, unique, or privileged operation.

from pandas import Series, date_range

s = Series([10, 200, 3_000], index=date_range('2020-01-01', periods=3))

print(
    s[2],  # label
    s[:'2020-01-01'], # positional
    sep='\n',
)
from pandas import Series

s = Series([10, 200, 3_000], index=[0, 1, 2])

print(
    s.loc[0],
    s.loc[:1],
    s.iloc[0],
    s.iloc[:1],
    sep='\n',
)
Similarly, consider len on a pandas.DataFrame.

from pandas import DataFrame, date_range
from numpy.random import default_rng

rng = default_rng(0)

df = DataFrame(
    index=(idx := date_range('2020-01-01', periods=3)),
    data={
        'a': rng.normal(size=len(idx)),
        'b': rng.integers(-10, +10, size=len(idx)),
    },
)

for x in df.columns:
    print(f'{x = }')

print(
    df,
    # f'{len(df) = }',
    # f'{len(df.index) = }',
    # f'{len(df.columns) = }',
    # f'{df.size  = }',
    # f'{df.shape = }',
    sep='\n{}\n'.format('\N{box drawings light horizontal}' * 20),
)
Where we break this intuition, we can see how it can impede understandability.

For example, when reviewing code, what transformations are safe? If we rely on assumptions of how __getitem__ typically works, a transformation such as the below should be fine:

from dataclasses import dataclass, field
from random import Random

@dataclass
class T(dict):
    random_state : Random = field(default_factory=Random)

    def __missing__(self, k):
        return self.random_state.random()

def f(x, y): pass
def g(x): pass

obj = T(random_state=Random(0))
k = ...
# f(obj[k], g(obj[k]))
v = obj[k]
f(v, g(v))
However, consider __dict__.__or__ which breaks a mathematical assumption of commutativity. Does this impede understandability?

d0 = {'a': 1,  'b': 2,  'c': 3, 'd': 4}
d1 = {                  'c': 30, 'd': 40}

print(
    f'{d0 | d1 = }',
    f'{d1 | d0 = }',
    sep='\n',
)
Of course…

s0 = {True}
s1 = {1}

print(
    f'{s0 | s1 = }',
    f'{s1 | s0 = }',
    f'{s0 == s1 = }',
    sep='\n',
)
… and also…

s0 = 'abc'
s1 = 'def'

print(
    f'{s0 + s1 = }',
    f'{s1 + s0 = }',
    sep='\n',
)
Does this make things more understandable?