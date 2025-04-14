What is the difference between mutable and immutable data… and how can I use this to improve my code?
print("Let's take a look!")
Obviously, mutable is data that we can change and immutable data is data that we cannot change. However, an important qualifier is whether we can change data in place.

s = 'abc'

print(f'before {s = } {id(s) = :#_x}')
s = s.upper()
print(f'after  {s = } {id(s) = :#_x}')

xs = [1, 2, 3]

print(f'before {xs = } {id(xs) = :#_x}')
xs.append(4)
print(f'after  {xs = } {id(xs) = :#_x}')
In both cases, the values changed, but only for xs (a mutable list) did the value change in place. If we captured a reference to the list in another name, we would be able to observe this change in two places.

s0 = s1 = 'abc'
xs0 = xs1 = [1, 2, 3]

print(
    f'before {s0 = } · {xs0 = }',
    f'       {s1 = } · {xs1 = }',
    sep='\n',
)

s0 = s0.upper()
xs0.append(4)

print(
    f'after  {s0 = } · {xs0 = }',
    f'       {s1 = } · {xs1 = }',
    sep='\n',
)
We can litigate the mechanisms used to enforce mutability, and there are many choices. However, while the exact mechanism may have some performance or some narrow correctness consequences, it is largely irrelevant to our purposes. (Recall that the “real world” appears to be fundamentally mutable.)

t = 'abc', 123
# t[0] = ...

class T:
    def __init__(self, x):
        self._x = x
    @property
    def x(self):
        return self._x

obj = T(123)
# obj.x = ...
obj._x = ...
Mutability allows us to have “action at a distance”: a change in one part of the code can change some other, non-local part of the code.

from threading import Thread
from time import sleep

class T:
    def __init__(self, values):
        self.values = values.copy()

    def __call__(self):
        while True:
            sleep(1)
            self.values.append(sum(self.values))

values = [1, 2, 3]
Thread(target=T(values)).start()

for _ in range(3):
    print(f'{values = }')
    sleep(1)
This can readily lead to code that is hard to understand using only local information.

One way to avoid this is to aggressively make copies any time we pass data around. However, we will have to be careful to make “deep copies.”

from threading import Thread
from time import sleep

class T:
    def __init__(self, values):
        self.values = values.copy()

    def __call__(self):
        while True:
            sleep(1)
            self.values[-1].append(sum(self.values[-1]))

values = [1, 2, 3, [4]]
Thread(target=T(values)).start()

for _ in range(3):
    print(f'{values = }')
    sleep(1)
Note that just as there is a distinction between a “deep” and a “shallow” copy, we can make a distinction between a “shallowly” and “deeply” immutable structure.

t = 'abc', [0, 1, 2]

print(f'before {t = }')
t[-1].append(3)
print(f'after  {t = }')
Alternatively, we could design around immutable data structures, using mechanisms such as a collections.namedtuple or dataclasses.dataclass. This can help us ensure that we do not inadvertantly mutate data non-locally. Of course, we will still have to be careful if these structures are only “shallowly” immutable.

from collections import namedtuple
from dataclasses import dataclass

@dataclass(frozen=True)
class T:
    value : int
obj = T(123)

T = namedtuple('T', 'value')
obj = T(123)
When we want to change our data, we will use mechanisms such as ._replace or dataclasses.replace() to replace and copy the entities as a whole.

from collections import namedtuple
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class T:
    value : int
obj0 = obj1 = T(123)
obj2 = replace(obj0, value=obj0.value * 10)
print(f'{obj0 = } · {obj1 = } · {obj2 = }')

T = namedtuple('T', 'value')
obj0 = obj1 = T(123)
obj2 = obj0._replace(value=obj0.value * 10)
print(f'{obj0 = } · {obj1 = } · {obj2 = }')
Note that we can keep references to the parts of the data that did not change, and we can rely on the Python garbage collector to keep those references alive only as long as they are needed. As a consequence, we may not necessarily see significantly increased memory usage from these copies.

We can use other tricks, like a collections.ChainMap, to reduce the amount of copied information (though at the loss of functionality, such as the ability to del an entry.)

from dataclasses import dataclass, replace, field
from collections import ChainMap
from random import Random
from string import ascii_lowercase

@dataclass(frozen=True)
class T:
    values : ChainMap[dict[str, int]] = field(default_factory=ChainMap)
    def __call__(self, *, random_state=None):
        rnd = random_state if random_state is not None else Random()
        new_entries = {
            ''.join(rnd.choices(ascii_lowercase, k=4)): rnd.randint(-100, +100)
            for _ in range(10)
        }
        return replace(self, values=ChainMap(new_entries, self.values))
    def __getitem__(self, key):
        return self.values[key]

rnd = Random(0)
obj = T()
for _ in range(3):
    obj = obj(random_state=rnd)

print(
    f'{obj = }',
    f'{obj["fudo"] = }',
    sep='\n{}\n'.format('\N{box drawings light horizontal}' * 40),
)
However, some very useful parts of Python are inherently mutable. For example, a generator or generator coroutine cannot be copied—at most, we can tee them, and that may not even necessarily work or be meanignful. (Of course, for many generators and generator coroutines, mutations may not be particularly problematic.)

Additionally, with a strictly immutable design, we have to be very clear about how the parts of our code share state. If we do not design two parts of our code to share state upfront, we may later discover that it is very disruptive to thread that state through later.

from dataclasses import dataclass
from functools import wraps
from itertools import count
from threading import Thread
from time import sleep
from typing import Iterator

@dataclass
class T:
    it : Iterator
    def __call__(self):
        while True:
            next(self.it)
            # self.it.send(True)
            sleep(1)

@lambda coro: wraps(coro)(lambda *a, **kw: [ci := coro(*a, **kw), next(ci)][0])
def resettable_count(start=0):
    while True:
        for state in count():
            if (reset := (yield start + state)):
                break
            # from inspect import currentframe, getouterframes
            # print(f'{getouterframes(currentframe())[1].lineno = }')

rc = resettable_count(start=100)
print(f'{rc.send(True) = }')
print(f'{next(rc)      = }')
Thread(target=(obj := T(rc))).start()
print(f'{next(rc)      = }')
print(f'{next(rc)      = }')
print(f'{rc.send(True) = }')
print(f'{next(rc)      = }')
How can I use this to improve my code?

What is the difference between immutability and hashability… and how does this affect my design?
print("Let's take a look!")
We know that the keys of a dict and the elements of a set must be hashable.

# hashable → ok ✓
d = {'a': ..., 'b': ..., 'c': ...}
s = {'a', 'b', 'c'}

# hashable → ok ✓
d = {('a', 'b', 'c'): ...}
s = {('a', 'b', 'c')}

# not hashable → not ok ✗
# d = {['a', 'b', 'c']: ...)
# s = {['a', 'b', 'c']}
This leads to clumsiness such as not being able to model set[set]—“sets of sets.” Since set is not hashable, we cannot create a set that contains another set. However, we can create set[frozenset]—“sets of frozensets.”

# s = a            # not ok ✗
s = {frozenset({'a', 'b', 'c'})} # ok ✓
Similarly, the keys of a dict can be frozenset but not set.

d = {
    frozenset({'a', 'b', 'c'}): ...
}
d[frozenset({'a', 'b', 'c'})]
This may be useful in cases where we want a compound key that has unique components where order does not matter.

d = {
    'a,b,c': ...,
}
print(f"{d['a,b,c'] = }")
for k in d:
    k.split(',')

d = {
    ('a', 'b', 'c', 'd,e'): ...
}
print(f"{d['a', 'b', 'c', 'd,e'] = }")
for x, y, z, w in d:
    pass

d = {
    frozenset({'a', 'b', 'c'}): ...
}
print(f"{d[frozenset({'a', 'b', 'c'})] = }")
print(f"{d[frozenset({'c', 'b', 'a'})] = }")
for k in d:
    pass
Naïvely, we may assume that the difference between set and frozenset that leads to frozenset being hashable is immutability. We may naïvely (and incorrectly) assert that hashability implies immutability (and vice-versa.)

In fact, for many of the common built-in types, we will see that those that are immutable are hashable and those that are mutable are not hashable.

xs = [1, 2, 3]            # `list`        mutable; not hashable
s  = {1, 2, 3}            # `set`         mutable; not hashable
d  = {'a': 1}             # `dict`        mutable; not hashable
t  =  'a', 1              # `tuple`     immutable;  is hashable
s  = frozenset({1, 2, 3}) # `frozenset` immutable;  is hashable

x = 12, 3.4, 5+6j, False # `int`, `float`, `complex`, bool` immutable; is hashable
x = 'abc', b'def'        # `str`, `bytes`                   immutable; is hashable

x = range(10) # `range` immutable; is hashable
When we discover that slice is immutable not hashable, we may chalk this up to a corner-case driven by syntactical ambiguity. (In fact, in later versions slice becomes hashable.)

x = slice(None)

# x.start = 0 # AttributeError

hash(x) # TypeError
In may be ambiguous to __getitem__ with a slice, since you cannot distinguish between a single-item lookup where that item is a slice and a multi-item sliced-lookup. In the case of builtin dict (which does not support multi-item) lookup, this isn’t much of a problem; however, note that pandas.Series.loc supports both modalities.

d = {
    slice(None): ...
}

print(
    f'{d[slice(None)] = }',
    f'{d[:] = }',
    sep='\n',
)
from pandas import Series

s = Series({
    None:        ...,
    slice(None): ...,
})

print(
    f'{s.loc[None]}',
    f'{s.loc[slice(None)]}',
    f'{s.loc[:]}',
    sep='\n{}\n'.format('\N{box drawings light horizontal}' * 40),
)
Additionally, since we can implement __hash__, we can create mutable objects that are hashable. Again, we may assume that this does not materially affect the relationship between hashability and immutability.

from dataclasses import dataclass

@dataclass
class T:
    value : list[int]
    def __hash__(self):
        return hash(id(self))

obj = T([1, 2, 3])
print(f'{hash(obj) = :#_x}')
obj.value.append(4)
print(f'{hash(obj) = :#_x}')
However, if we consider more deeply the relationship between the two, we will discover the true nature of mutability and hashability.

Let’s consider two different ways to compute the hash of a piece of data:

on its identity
on its value (equality)
class Base:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f'T({self.value!r})'

class T0(Base):
    def __hash__(self):
        return hash(id(self))

class T1(Base):
    def __hash__(self):
        return hash(self.value)

obj0, obj1 = T0(123), T1(123)

print(
    f'{hash(obj0) = }',
    f'{hash(obj1) = }',
    sep='\n',
)
Note that the hash when computed on identity changes across runs. In general, since the underlying mechanism of hash is an internal implementation detail, hash values may readily change across versions of Python.

from random import Random

rnd = Random(0)
x = (
    rnd.random(),
    rnd.random(),
)

print(
    f'x =       {     x }',
    f'hash(x) = {hash(x)}',
    sep='\n',
)
Assume that the value is immutable. If we were to compute the hash based on identity, then we might accidentally “lose” an object in a dict.

from dataclasses import dataclass

@dataclass(frozen=True)
class T:
    value : int
    def __hash__(self):
        return hash(id(self))

def f(d):
    d[obj := T(123)] = ...

d = {}
f(d)

# d[T(123)] # KeyError
for k in d:
    print(f'{k = }')
Therefore, we must hash immutable objects based on their value (on equality.) This is a matter of practicality.

Assume that the value is mutable. If we were to compute the hash based on equality, then we might accidentally “lose” an object in a dict.

from dataclasses import dataclass

@dataclass
class T:
    value : int
    def __hash__(self):
        return hash(self.value)

d = {}
d[obj := T(123)] = ...
obj.value = 456

# d[obj] # KeyError
# d[T(123)] # KeyError
# d[T(456)] # KeyError
# for k in d: print(f'{k = }')
This is a serious problem, because the hash that was used to determine the location of the entry in the dict is no longer accurate. There will be no way to retrieve the value via __getitem__!

Therefore, we must hash mutable objects based on their identity. However, we still have the problem of “losing” a value in the dict if we hash on identity.

Except the value is still in the dict; we simply cannot access it via __getitem__. We can still iterate over dict in both cases!

from dataclasses import dataclass

@dataclass
class T:
    value : int
    def __hash__(self):
        return hash(self.value)

d = {}
d[obj := T(123)] = ...
obj.value = 456

for k in d:
    print(f'{k = }')
We may then extend our understanding of this topic as follows: immutable objects must be hashed on value to support direct retrieval with equivalent objects; mutable objects must be hashed on identity and cannot support direct retrieval. In other words, hashability implies immutability if-and-only-if we need direct (or “non-intermediated” access.)

In fact, it is relatively common to see hashed mutable object. Consider the use of a networkx.DiGraph with a custom, rich node type. (Our Node class must be hashable, since the networkx.DiGraph is implemented as a “dict of dict of dicts.”)

from dataclasses import dataclass
from itertools import pairwise

from networkx import DiGraph

@dataclass
class Node:
    name  : str
    value : int = 0
    def __hash__(self):
        return hash(id(self))

nodes = [Node('a'), Node('b'), Node('c')]

g = DiGraph()
g.add_edges_from(pairwise(nodes))

for n in nodes:
    n.value += 1

for n in g.nodes:
    ...
Consider, however, that all access to the nodes of the networkx.DiGraph will likely be intermediated by calls such as .nodes that allow us to iterate over all of the nodes. We may also subclass networkx.DiGraph to allow direct access to nodes by name, further intermediating between the __getitem__ syntax and the hash-lookup mechanism.

from dataclasses import dataclass
from itertools import pairwise

from networkx import DiGraph

@dataclass
class Node:
    name  : str
    value : int = 0
    def __hash__(self):
        return hash(id(self))

nodes = [Node('a'), Node('b'), Node('c')]

class MyDiGraph(DiGraph):
    class by_name_meta(type):
        def __get__(self, instance, owner):
            return self(instance)

    @dataclass
    class by_name(metaclass=by_name_meta):
        instance : 'MyDiGraph'
        def __getitem__(self, key):
            nodes_by_name = {k.name: k for k in self.instance.nodes}
            return nodes_by_name[key]

g = MyDiGraph()
g.add_edges_from(pairwise(nodes))

for n in nodes:
    n.value += 1

print(f"{g.by_name['a'] = }")
Note that it is not a good idea to store object id(…)s in structures, since (in CPython) the memory addresses for these objects (and their corresponding id(…) values) may be reüsed. However, over the lifetime of an object, its id(…) will not change, so it is safe to store the id(…) if the lifetime of this storage is tied to the lifetime of the object. This will be the case with hashing an object on id(…) and putting it into a set or dict. While the __hash__(…) will be implicitly stored and is a dependent value of id(…), the lifetime of that storage will necessarily match to the lifetime of the object itself. Furthermore, the hash is used only to find the approximate location of the entry in the set or dict. Since hash values are finite (in CPython, constrained to the valid range of Py_hash_t values where Py_hash_t is typedefd to Py_ssize_t which is generally typedefd to ssize_t,) then by the “pigeonhole principle,” multiple distinct objects must share the same hash. Therefore, after performing any necessary additional “probing,” the set or dict will perform an == comparison to confirm that it has found the right item. This further ensures that computing __hash__ on id(…) won’t lead to stale entries.

It also means that objects which are not equivalent to themselves trivially get lost in dicts! For example, float('nan') can be the key of a dict, but you will not be able to later retrieve the value via direct __getitem__!

d = {
    float('nan'): ...,
}

d[float('nan')] # KeyError
How does this affect my design?