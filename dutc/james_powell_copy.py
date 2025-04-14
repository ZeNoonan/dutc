What is the difference between a live view and a snapshot… and why does it matter?
print("Let's take a look!")
A “snapshot (copy)” is a static copy of some state at some point in time; a “live view” is a dynamic reference to some state.

xs = [1, 2, 3]

xs.append(4)
ys = xs.copy()
xs.append(5)

print(
    f'{xs = }',
    f'{ys = }',
    sep='\n',
)
Whereas…

xs = [1, 2, 3]
ys = xs

xs.append(4)
xs.append(5)

print(
    f'{xs = }',
    f'{ys = }',
    sep='\n',
)
We may desire a “live view” to eliminate “update anomalies”: cases where an update to one part of the system should be reflected in another part of the system, cases where we want a “single source of truth.”

from dataclasses import dataclass
from copy import copy

@dataclass
class Employee:
    name   : str
    role   : str
    salary : float

@dataclass
class Entitlement:
    employee : Employee
    access   : bool

employees = {
    'alice': Employee('alice', 'programmer', 250_000),
    'bob':   Employee('bob',   'programmer', 225_000),
}

entitlements = {
    k: Entitlement(employee=v, access=False)
    for k, v in employees.items()
}

payroll_by_year = {
    2020: {
        k: copy(v) for k, v in employees.items()
    },
}

employees['alice'].role = 'manager'
employees['alice'].salary *= 1.5

print(
    f'{employees["alice"].role             = }',
    f'{entitlements["alice"].employee.role = }',
    f'{payroll_by_year[2020]["alice"].role = }',
    sep='\n',
)
Copies can be made explicitly or implicitly in a number of different ways.

from copy import copy

xs = [1, 2, 3]
# ys = xs
ys = [*xs]
# ys = list(xs)
# ys = xs.copy()
# ys = copy(xs)

xs.append(4)

print(
    f'{xs = }',
    f'{ys = }',
    sep='\n',
)
We often want to distinguish between “shallow” and “deep” copies. A “shallow copy” is a copy of only the top “level” of a nested container structure. A “deep copy” copies all levels of the nested structure.

xs = [
    [1, 2, 3],
    [4, 5, 6, 7],
]
ys = xs.copy() # or `copy.copy(xs)`

xs[0].insert(0, 0)
xs.append([8, 9])

print(
    f'{xs = }',
    f'{ys = }',
    sep='\n',
)
Whereas with a copy.deepcopy…

from copy import deepcopy

xs = [
    [1, 2, 3],
    [4, 5, 6, 7],
]
ys = deepcopy(xs)

xs[0].insert(0, 0)
xs.append([8, 9])

print(
    f'{xs = }',
    f'{ys = }',
    sep='\n',
)
Given the two changes made to xs, we can distinguish between:

a “live view” that observes both changes
a “shallow copy” that observes only the deeper change
a “deep copy” that observes neither changes
(There is a necessary asymmetry here: we cannot observe only the shallow change but not the deeper change.)

from copy import copy, deepcopy

xs = [
    [1, 2, 3],
    [4, 5, 6, 7],
]

ys = {
    # i.    ii.
    (True, True):   xs,
    (True, False):  copy(xs),
    # (False, True):  ...,
    (False, False): deepcopy(xs),
}

xs[0].insert(0, 0) # i.
xs.append([8, 9])  # ii.

print(
    f'{xs = }',
    *ys.values(),
    sep='\n',
)
Clearly, we want a “snapshot” if we want to capture the state as of a certain point in time and not observe later updates (i.e., mutations.) We want a “live view” view if we do want to see later updates.

The .keys() on a dict (which used to be called .viewkeys() in Python 2,) is a live view of the keys of a dict. As a consequence, if we capture a reference to it, then subsequently mutate the dict, we will see that mutation when iterating over the reference we have captured.

d = {'abc': 123, 'def': 456, 'xyz': 789}

keys = d.keys() # “live view”

d['ghi'] = 999

for k in keys:
    print(f'{k = }')
However, if we wanted a snapshot, we may need to explicitly trigger a copy.

d = {'abc': 123, 'def': 456, 'xyz': 789}

keys = [*d.keys()] # “snapshot”

d['ghi'] = 999

for k in keys:
    print(f'{k = }')
Similarly, we can consider the different import styles to be an instance of “early”- vs “late”-binding, which is similar phraseology around the idea of “snapshots” vs “live views.”

from textwrap import dedent
from math import cos, sin, pi

print(
    f'before {sin(pi) = :>2.0f}',
    f'       {cos(pi) = :>2.0f}',
    sep='\n',
)

# don't “pollute” namespace
exec(dedent('''
    import math
    math.sin, math.cos = math.cos, math.sin
'''))

print(
    f'after  {sin(pi) = :>2.0f}',
    f'       {cos(pi) = :>2.0f}',
    sep='\n',
)
However…

from textwrap import dedent
import math

print(
    f'before {math.sin(math.pi) = :>2.0f}',
    f'       {math.cos(math.pi) = :>2.0f}',
    sep='\n',
)

# don't “pollute” namespace
exec(dedent('''
    import math
    math.sin, math.cos = math.cos, math.sin
'''))

print(
    f'after  {math.sin(math.pi) = :>2.0f}',
    f'       {math.cos(math.pi) = :>2.0f}',
    sep='\n',
)
In fact, we can think of dotted __getattr__ lookup as being a key mechanism in getting a “live view” of some data.

from dataclasses import dataclass

@dataclass
class T:
    x : int

obj = T(123)
x = obj.x

print(f'before {obj.x = } · {x = }')
obj.x = 456
print(f'after  {obj.x = } · {x = }')
There are many subtle design distinctions we can make in our code that differ in terms of whether they provide us with a “live view“ or a “snapshot.”

These four variations have some subtle distinctions:

class Base:
    def __repr__(self):
        return f'{type(self).__name__}({self.values!r})'

# i.
class T1(Base):
    def __init__(self, values):
        self.values = values

# ii.
class T2(Base):
    def __init__(self, values):
        self.values = [*values]

# iii.
class T3(Base):
    def __init__(self, values):
        self.values = values.copy()

# iv.
class T4(Base):
    def __init__(self, *values):
        self.values = values

values = [1, 2, 3]
obj = T1(values)
values.clear()
print(f'i.   {values = } · {obj = }')

values = [1, 2, 3]
obj = T2(values)
values.clear()
print(f'ii.  {values = } · {obj = }')

values = [1, 2, 3]
obj = T3(values)
values.clear()
print(f'iii. {values = } · {obj = }')

values = [1, 2, 3]
obj = T4(*values)
values.clear()
print(f'iv.  {values = } · {obj = }')
However, this is not the only distinction between the above!

from collections import deque

class Base:
    def __repr__(self):
        return f'{type(self).__name__}({self.values!r})'

# i.
class T1(Base):
    def __init__(self, values):
        self.values = values

# ii.
class T2(Base):
    def __init__(self, values):
        self.values = [*values]

# iii.
class T3(Base):
    def __init__(self, values):
        self.values = values.copy()

# iv.
class T4(Base):
    def __init__(self, *values):
        self.values = values

values = deque([1, 2, 3], maxlen=3)
obj = T1(values)
values.append(4)
print(f'i.   {values = } · {obj = }')

values = deque([1, 2, 3], maxlen=3)
obj = T2(values)
values.append(4)
print(f'ii.  {values = } · {obj = }')

values = deque([1, 2, 3], maxlen=3)
obj = T3(values)
values.append(4)
print(f'iii. {values = } · {obj = }')

values = deque([1, 2, 3], maxlen=3)
obj = T4(*values)
values.append(4)
print(f'iv.  {values = } · {obj = }')
We can think of “inheritance” as a mechanism for “live updates.”

class Base:
    pass

class Derived(Base):
    pass

Base.f = lambda _: ...

print(
    f'{Derived.f = }',
    sep='\n',
)
In fact, if we extend the idea of changes to changes across versions of our code, we can see a material distinction between “inheritance,” “composition,” and alternate approaches.

class Base:
    def f(self):
        pass

# statically added (e.g., in a later version)
Base.g = lambda _: ...

class Derived(Base):
    pass

class Composed:
    def __init__(self, base : Base = None):
        self.base = Base() if base is None else base
    def f(self, *args, **kwargs):
        return self.base.f(*args, **kwargs)

class Constructed:
    locals().update(Base.__dict__)

    ### alternatively…
    # f = Base.f
    # g = Base.g

# dynamically added (e.g., via monkey-patching)
Base.h = lambda _: ...

print(
    ' Derived '.center(40, '\N{box drawings light horizontal}'),
    f'{hasattr(Derived,     "f") = }',
    f'{hasattr(Derived,     "g") = }',
    f'{hasattr(Derived,     "h") = }',
    ' Composed '.center(40, '\N{box drawings light horizontal}'),
    f'{hasattr(Composed,    "f") = }',
    f'{hasattr(Composed,    "g") = }',
    f'{hasattr(Composed,    "h") = }',
    ' Constructed '.center(40, '\N{box drawings light horizontal}'),
    f'{hasattr(Constructed, "f") = }',
    f'{hasattr(Constructed, "g") = }',
    f'{hasattr(Constructed, "h") = }',
    sep='\n',
)
Consider the collections.ChainMap, which allows us to isolate writes to the top “level” of a multi-level structure. This mechanism is closely related to how both scopes and how __getattr__ and __setattr__ work.

base = {
    'abc': 123
}

snapshot = {
    **base,
    'def': 456,
}

# base['abc'] *= 2
# snapshot['abc'] *= 10

print(
    f'{base     = }',
    f'{snapshot = }',
    sep='\n',
)
from collections import ChainMap

base = {
    'abc': 123
}

layer = {
    'def': 456,
}

live = ChainMap(layer, base)

# live['abc'] *= 10
base['abc'] *= 2

print(
    f'{base = }',
    f'{live = } · {live["abc"] = }',
    sep='\n',
)
It is important that we be aware of “shadowing” where something that may appear to be a “live view” may become a “snapshot.”

Recall the subtle distinction between clearing a list via the following approaches. If we have captured a “live view” of xs with ys, then we must mutate xs with .clear() or del xs[:] for the clearing to be visible on ys.

# i.
xs = ys = [1, 2, 3]
xs = []
print(f'{xs = } · {ys = }')

# ii.
xs = ys = [1, 2, 3]
xs.clear()
print(f'{xs = } · {ys = }')

# iii.
xs = ys = [1, 2, 3]
del xs[:]
print(f'{xs = } · {ys = }')
Similarly, manipulating sys.path requires that we manipulate the actual sys.path. A name binding of path = … in the module scope doesn’t change the actual sys.path.

from tempfile import TemporaryDirectory
from pathlib import Path

with TemporaryDirectory() as d:
    d = Path(d)
    with open(d / '_module.py', mode='w') as f:
        pass

    from sys import path
    path.append(f'{d}') # works!
    from sys import path
    path.insert(0, f'{d}') # works!

    from sys import path
    path = path + [f'{d}'] # does not work!
    from sys import path
    path = [f'{d}'] +path # does not work!

    import sys
    sys.path.append(f'{d}') # works!
    import sys
    sys.path.insert(0, f'{d}') # works!

    import sys
    sys.path = sys.path + [f'{d}'] # works!
    # what about [*sys.path, f'{d}']… ?
    import sys
    sys.path = [f'{d}'] + sys.path # works!
“Shadowing” is how we can describe what happens when we create a “shadow” (“snapshot (copy)”) of some data at some higher level of a scoped lookup. This can easily happen in our OO hierarchies if we are not careful.

class Base:
    x = []

class Derived(Base):
    pass

# Base.x.append(1)
# Derived.x.append(2)
# Base.x = [1, 2, 3, 4]
Derived.x = [1, 2, 3, 4, 5, 6]

print(
    f'{Base.x                  = }',
    f'{Derived.x               = }',
    f'{Base.__dict__.keys()    = }',
    f'{Derived.__dict__.keys() = }',
    sep='\n',
)
But… what if the value is immutable? If the value is immutable, then we have to be particularly careful to update it at the right level!

class Base:
    x = 123

class Derived(Base):
    pass

Derived.x = 789
Base.x = 456
# del Derived.x

print(
    f'{Base.x                  = }',
    f'{Derived.x               = }',
    f'{Base.__dict__.keys()    = }',
    f'{Derived.__dict__.keys() = }',
    sep='\n',
)