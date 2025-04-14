What is boilerplate elimination, is boilerplate really that bad… and how can eliminating it help me work faster?
print("Let's take a look!")
In order to actually make a class-style object in Python useful, we need to write an lot of “boilerplate.”

class T:
    def __init__(self, value):
        self._value = value
    @property
    def value(self):
        return self._value
    def __hash__(self):
        return hash(self.value)
    def __eq__(self, other):
        return self.value == other.value
    def __repr__(self):
        return f'T({self.value!r})'

obj0, obj1 = T(123), T(123)
print(
    f'{obj0.value     = }',
    f'{obj1.value     = }',
    f'{obj0 == obj1   = }',
    f'{({obj0, obj1}) = }',
    sep='\n',
)
We can reduce this boilerplate in a couple of ways. One way is the use of a collections.namedtuple:

from collections import namedtuple

T = namedtuple('T', 'value')

obj0, obj1 = T(123), T(123)
print(
    f'{obj0.value     = }',
    f'{obj1.value     = }',
    f'{obj0 == obj1   = }',
    f'{({obj0, obj1}) = }',
    sep='\n',
)
Another option is a dataclasses.dataclass:

from dataclasses import dataclass

@dataclass(frozen=True)
class T:
    value : int

obj0, obj1 = T(123), T(123)
print(
    f'{obj0.value     = }',
    f'{obj1.value     = }',
    f'{obj0 == obj1   = }',
    f'{({obj0, obj1}) = }',
    sep='\n',
)
However, beyond just the reduction in lines-of-code, consider the “escalation pathway” we are provided with these:

entities = [
    ('abc', 123),
    ('def', 456),
    ('xyz', 789),
]

...
...
...

for ent in entities:
    print(f'{ent[0].upper() = }', f'{ent[1] + 1 = }', sep='\N{middle dot}'.center(3))

for name, value in entities:
    print(f'{name.upper() = }', f'{value + 1 = }', sep='\N{middle dot}'.center(3))
Using a list[tuple] is a very simple and quick way to start our programme, but as our code grows, the poor ergonomics show themselves quickly.

It is at this point we may “graduate” the code to use a collections.namedtuple.

We may first create the new collections.namedtuple type:

from collections import namedtuple

Entity = namedtuple('Entity', 'name value')
Then we may apply it to our existing data:

from collections import namedtuple

Entity = namedtuple('Entity', 'name value')

entities = [
    Entity('abc', 123),
    Entity('def', 456),
    Entity('xyz', 789),
]

for ent in entities:
    print(f'{ent[0].upper() = }', f'{ent[1] + 1 = }', sep='\N{middle dot}'.center(3))

for name, value in entities:
    print(f'{name.upper() = }', f'{value + 1 = }', sep='\N{middle dot}'.center(3))
Then we may rewrite any code that uses unpacking or indexing syntax to use __getattr__ (named-lookup) syntax:

from collections import namedtuple

Entity = namedtuple('Entity', 'name value')

entities = [
    Entity('abc', 123),
    Entity('def', 456),
    Entity('xyz', 789),
]

# for ent in entities:
#     print(f'{ent[0].upper() = }', f'{ent[1] + 1 = }', sep='\N{middle dot}'.center(3))

# for name, value in entities:
#     print(f'{name.upper() = }', f'{value + 1 = }', sep='\N{middle dot}'.center(3))

for ent in entities:
    print(f'{ent.name.upper() = }', f'{ent.value + 1 = }', sep='\N{middle dot}'.center(3))
This allows to add fields:

from collections import namedtuple

Entity = namedtuple('Entity', 'name value flag')

entities = [
    Entity('abc', 123, True),
    Entity('def', 456, False),
    Entity('xyz', 789, True),
]

for ent in entities:
    print(f'{ent.name.upper() = }', f'{ent.value + 1 = }', sep='\N{middle dot}'.center(3))
We may subclass the collections.namedtuple to support validation and defaults:

from collections import namedtuple

class Entity(namedtuple('EntityBase', 'name value flag')):
    def __new__(cls, name, value, flag=False):
        if value < 0:
            raise ValueError('value should not be negative')
        return super().__new__(cls, name, value, flag)

entities = [
    Entity('abc', 123),
    Entity('def', 456),
    Entity('xyz', 789, flag=True),
]

for ent in entities:
    print(
        f'{ent.name.upper() = }',
        f'{ent.value + 1 = }',
        f'{ent.flag = }',
        sep='\N{middle dot}'.center(3),
    )
We may further raise this into a dataclasses.dataclass if we need to add instance methods, to add additional protocols, to customise protocol implementation, or to support mutability.

from dataclasses import dataclass

@dataclass
class Entity:
    name  : str
    value : int
    flag  : bool = False
    def __post_init__(self):
        if self.value < 0:
            raise ValueError('value should not be negative')
    def __call__(self):
        self.value += 1
    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

entities = [
    Entity('abc', 123),
    Entity('def', 456),
    Entity('xyz', 789, flag=True),
]

for ent in entities:
    ent()
    print(
        f'{ent.name.upper() = }',
        f'{ent.value + 1 = }',
        f'{ent.flag = }',
        sep='\N{middle dot}'.center(3),
    )
Finally, we may rewrite as a class-style object with all of the boilerplate.

class Entity:
    def __init__(self, name, value, flag=False):
        if value < 0:
            raise ValueError('value should not be negative')
        self.name, self.value, self.flag = name, value, flag
    def __call__(self):
        self.value += 1
    def __eq__(self, other):
        return self.name == other.name and self.value == other.value
    def __repr__(self):
        return f'Entity({self.name!r}, {self.value!r}, {self.flag!r})'

entities = [
    Entity('abc', 123),
    Entity('def', 456),
    Entity('xyz', 789, flag=True),
]

for ent in entities:
    ent()
    print(
        f'{ent.name.upper() = }',
        f'{ent.value + 1 = }',
        f'{ent.flag = }',
        sep='\N{middle dot}'.center(3),
    )
There are other boilerplate-elimination tools in the Python standard library.

For example, enum.Enum allows us to create enumerated types easily.

from enum import Enum

Choice = Enum('Choice', 'A B C')

print(
    f'{Choice.A = }',
    f'{Choice.B = }',
    f'{Choice.C = }',
    sep='\n',
)
functools.total_ordering allows us to implement comparison operators without having to write them all out (assuming the object supports mathematical properties associated with a total ordering.)

from dataclasses import dataclass
from functools import total_ordering

@total_ordering
@dataclass
class T:
    value : int
    def __eq__(self, other):
        return self.value == other.value
    def __lt__(self, other):
        return self.value < other.value
    # def __gt__(self, other):
    #     return self.value > other.value
    # def __ne__(self, other):
    #     return self.value != other.value
    # def __lte__(self, other):
    #     return self.value <= other.value
    # def __gte__(self, other):
    #     return self.value >= other.value
A contextlib.contextmanager allows us to situate a generator into the contextmanager __enter__/__exit__ protocol.

class Context:
    def __enter__(self):
        print(f'T.__enter__')
    def __exit__(self, exc_value, exc_type, traceback):
        print(f'T.__exit__')

with Context():
    print('block')
from contextlib import contextmanager

@contextmanager
def context():
    print(f'__enter__')
    try: yield
    finally: pass
    print(f'__exit__')

with context():
    print('block')
How can eliminating it help me work faster?