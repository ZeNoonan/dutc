What is the difference between identity and equality… and why should I care?
print("Let's take a look!")
Python variable names are just that: names. They are names that we can use to refer to some underlying data.

The == operator in Python determines whether the objects referred to by two names are “equal.” For a container like list, this means that the two objects contain the same elements in the same order.

xs = [1, 20, 300]
ys = [1, 20, 300]
zs = [4_000, 50_000, 600_000]

print(
    f'{xs == ys = }',
    f'{xs == zs = }',
    sep='\n',
)
For a container like dict, this means that the two objects contain the same key value pairs; however, order is not considered.

d0 = {'a': 1, 'b': 20, 'c': 300}
d1 = {'c': 300, 'b': 20, 'a': 1}
d2 = {'d': 4_000, 'e': 5, 'f': 600_000}

print(
    f'{d0 is d1 = }',
    f'{d0 is d2 = }',
    sep='\n',
)
For a collections.OrderedDict, however, the order is considered when determinig equality.

from collections import OrderedDict

d0 = OrderedDict({'a': 1, 'b': 20, 'c': 300})
d1 = OrderedDict({'c': 300, 'b': 20, 'a': 1})

print(
    f'{d0 == d1 = }',
    sep='\n',
)
The is operator in Python determines whether the objects referred to by two names are, in fact, the same object. Unlike ==, this has consistent meaning irrespect of the type of the object.

You can specify what it means for two instances of a user-defined object to be equal (“equivalent”; ==,) but there is no way to specify an alternate or custom meaning for identity (is.)

from dataclasses import dataclass, field
from typing import Any

@dataclass
class T:
    name     : str
    value    : int
    metadata : dict[str, Any] = field(default_factory=dict)

    # do not consider `.metadata` for equality
    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

x = T('abc', 123)
y = T('abc', 123, metadata={...: ...})
z = T('def', 456)

print(
    f'{x == y = }',
    f'{x == z = }',
    sep='\n',
)
Similarly, while it is possible to overload many operators in Python, the assignment and assignment-expression operators (= and :=) cannot be customised in any fashion.

These operations are also called “name-bindings.”

x = y always means “x is a new name for the object that is currently referred to by y.” Unlike in other programming languages, x = y cannot directly trigger any other form of computation (e.g., a copy computation.)

However, since performing a name-binding sometimes involves assignment into a dict representing the active scope, the assignment into the dict can trigger other computations.

from collections.abc import MutableMapping
from logging import getLogger, basicConfig, INFO

logger = getLogger(__name__)
basicConfig(level=INFO)

class namespace(dict, MutableMapping):
    def __setitem__(self, key, value):
        logger.info('namespace.__setitem__(key = %r, value = %r)', key, value)
        super().__setitem__(key, value)

class TMeta(type):
    @staticmethod
    def __prepare__(name, bases, **kwds):
        return namespace()

class T(metaclass=TMeta):
    x = [1, 2, 3]
    y = x
An alternate way to determine whether two names refer to an identical object is to check their id(...) values. The id(...) return value is a (locally, temporally) unique identifier for an object. In current versions of CPython, this corresponds to the memory address of the PyObject* for the object (but this is not guaranteed.)

xs = [1, 20, 300]
ys = [1, 20, 300]
zs = xs

print(
    # f'{xs is ys = }',
    # f'{xs is zs = }',
    f'{id(xs) = :#_x}',
    f'{id(ys) = :#_x}',
    f'{id(zs) = :#_x}',
    sep='\n',
)
Another way to determine if two names refer to an identical object is to perform a mutation via one name and see whether the object referred to the other name has changed or not!

xs = [1, 20, 300]
ys = [1, 20, 300]
zs = xs

xs.append(4_000)

print(
    f'{xs = }',
    f'{ys = }',
    f'{zs = }',
    sep='\n',
)
Note that if two names refer to immutable objects, then those objects cannot be changed; therefore, we will not be able to observe a useful difference between these two names refering to identical objets or merely refering to equivalent objects. As a consequence, the CPython interpreter will try to save memory by “interning” commonly found immutable objects, such as short strings and small numbers. When “interning,” all instances of the same value are, in fact, instances of an identical object.

print(
    # f'{id(eval("123"))     = :#_x}',
    # f'{id(eval("123"))     = :#_x}',
    # f'{id(eval("123_456")) = :#_x}',
    # f'{id(eval("123_456")) = :#_x}',

    f'{id(123)             = :#_x}',
    f'{id(123)             = :#_x}',
    f'{id(123_456)         = :#_x}',
    f'{id(123_456)         = :#_x}',
    sep='\n',
)
We have to use eval in the above example, since (C)Python code in a script will undergo the “constant folding” optimisation.

from pathlib import Path
from sys import path
from tempfile import TemporaryDirectory
from textwrap import dedent

with TemporaryDirectory() as d:
    d = Path(d)
    with open(d / '_module.py', 'w') as f:
        print(dedent('''
            def h():
                x = 123_456_789
                y = 123_456_789
        ''').strip(), file=f)
    path.append(f'{d}')
    from _module import h

def f():
    x = 123_456_789
    y = 123_456_789

def g():
    x = 123_456_789
    y = 123_456_789

print(
    f'{f.__code__.co_consts = }',
    f'{g.__code__.co_consts = }',
    f'{h.__code__.co_consts = }',
    f'{f.__code__.co_consts[-1] is g.__code__.co_consts[-1] = }',
    f'{f.__code__.co_consts[-1] is h.__code__.co_consts[-1] = }',
    sep='\n',
)
The qualifications on “unique” are necessary. Recall that the CPython value for id(...) is currently implemented as the memory address of the object the name refers to (i.e., the value of the PyObject*.)

/* Python/bltinmodule.c */

static PyObject *
builtin_id(PyModuleDef *self, PyObject *v)
{
    PyObject *id = PyLong_FromVoidPtr(v);

    if (id && PySys_Audit("builtins.id", "O", id) < 0) {
        Py_DECREF(id);
        return NULL;
    }

    return id;
}
This can be used to do things we’re otherwise not supposed to, such as directly accessing Python objects.

from numpy import array
from numpy.lib.stride_tricks import as_strided

def setitem(t, i, v):
    xs = array([], dtype='uint64')
    if (loc := xs.__array_interface__['data'][0]) > id(t):
        raise ValueError('`numpy.ndarray` @ {id(xs):#_x} allocated after `tuple` @ {id(t):#_x}')
    xs  = as_strided(xs, strides=(1,), shape=((off := id(t) - loc) + 1,))
    ys  = as_strided(xs[off:], strides=(8,), shape=(4,))
    zs  = as_strided(ys[3:], strides=(8,), shape=(i + 1,))
    ys[2] += max(0, i - (end := len(t)) + 1)
    zs[min(i, end):] = id(v)

t = 0, 1, 2, None, 4, 5
print(f'Before: {t = !r:<24} {type(t) = }')
setitem(t, 3, 3)
print(f'After:  {t = !r:<24} {type(t) = }')
As a consequence of using the memory address as the value for id(…) coupled with the finiteness of memory, we would expect that memory addresses would eventually be reüsed. Therefore, across an arbitrary span of time, two objects with the same id(…) may, in fact, be distinct.

xs = [1, 2, 3]
print(f'{id(xs) = :#_x}')
del xs

ys = [1, 2, 3, 4]
print(f'{id(ys) = :#_x}')
We should not store id(…) values for comparison later. We may be tempted to do this in the case of unhashable objects, but the result will not be meaningful.

class T:
    def __hash__(self):
        raise NotImplementedError()

obj0, obj1 = T(), T()

print(
    # f'{obj0     in {obj0, obj1}         = }',
    # f'{id(obj0) in {id(obj0): obj0, id(obj1): obj1} = }',
    sep='\n',
)
(We see a very similar problem with child processes upon termination of the parent process; in general, since PID are a finite resource and may be reüsed, it is incorrect for us to store and refer to child processes across a span of time in the absence of some isolation mechanism such as a PID namespace.)

### (unsafely?) reduce maximum PID
# <<< $(( 2 ** 7 )) /proc/sys/kernel/pid_max

typeset -a pids=()

() { } & pids+=( "${!}" )

until
    () { } & pids+=( "${!}" )
    (( pids[1] == pids[${#pids}] ))
do :; done

printf 'pid[%d]=%d\n'     1      "${pids[1]}" \
                      "${#pids}" "${pids[${#pids}]}"
Therefore, the following code may be incorrect (since the PID we are killing may not necessarily be the process we think!)

sleep infinity & pid="${!}"

: ...

kill "${pid}"
Up to a name (re-)binding, equality is a transient property but identity is a permanent property. In other words, if two names refer to equal (“equivalent”) objects at some point in time, they may or may not remain equal at some later point in time. However, if two names refer to identical objects at some point in time, the only intervening action that can alter their identicalness is a name (re-)binding.

xs = [1, 2, 3]
ys = [1, 2, 3]

assert xs == ys
...
xs.clear()
...
assert xs != ys
xs = ys = [1, 2, 3]

assert xs is ys
...
# xs = ...
# (xs := ...)
# globals()['xs'] = ...
# from module import xs
...
assert xs is ys
Of course, if the two names refer to immutable objects, then their equivalence is also a permanent property!

Note that identity and equality are separate properties. Identicalness does not necessarily imply equivalence, nor does equivalence imply identiticalness.

# i. equal and identical
xs = ys = [1, 2, 3]
assert xs == ys and xs is ys

# ii. equal but not identical
xs, ys = [1, 2, 3], [1, 2, 3]
assert xs == ys and xs is not ys

# iii. identical and equal
x = y = 2.**53
assert x is y and x == y

# iv. identical but not equal
x = y = float('nan')
# x = y = None
class T:
    def __eq__(self, other):
        return False
assert x is y and x != y
However, note that if two names refer to identical objects, then we are guaranteed that the id(…) values (when captured at a single point in time during the lifetime of both objects) must have equivalent value.

x = y = object()

# two ways to state the same thing
assert x is y and id(x) == id(y)

# since id(…) returns an `int`,
#   since (in CPython) large `int`s are not interned,
#   since (in CPython) `id(…)` gives the memory address, and
#   since (in CPython) these memory addreses are in the upper ranges
#   the `int` that `id(x)` will be allocated separately than `int` that `int(y)`
#   returns, leading to the following…
assert x is y and id(x) is not id(y)
Since equality can be implemented via the object model (but identity cannot,) it is possible for an object to not be equivalent to even itself!

class T:
    def __eq__(self, other):
        return False

obj = T()
assert obj is obj and obj != obj
Note that since == can be implemented but is cannot, and that (in CPython) is is a pointer comparison, is checks are very likely to be consistently faster than == checks.

/* Include/object.h */

#define Py_Is(x, y) ((x) == (y))
Therefore, the use of an enum.Enum may prove faster than an equivalent string equality comparison in some cases. (Note, however, that object equality comparison may just as well implement an identity “fast-path,” minimising the performance improvement.

from time import perf_counter
from contextlib import contextmanager
from enum import Enum

@contextmanager
def timed(msg):
    before = perf_counter()
    try: yield
    finally: pass
    after = perf_counter()
    print(f'{msg:<48} \N{mathematical bold capital delta}t: {after - before:.6f}s')

def f(x):
    return x == 'abcdefg'

Choice = Enum('Choice', 'Abcdefg')
def g(x):
    return x is Choice.Abcdefg

with timed('f'):
    x = 'abcdefg'
    for _ in range(100_000):
        f(x)

with timed('g'):
    x = Choice.Abcdefg
    for _ in range(100_000):
        g(x)
Generally, whether two containers are equivalent is determined by checking whether their contents are equivalent.

def __eq__(self, other):
    if len(xs) != len(ys):
        return False
    for x, y in zip(xs, ys, strict=True):
        if x != y:
            return False
    return True

xs = [1, 2, 3]
ys = [1, 2, 3]

print(
    f'{xs == ys       = }',
    f'{__eq__(xs, ys) = }',
    sep='\n',
)
Except in the implementation of list, there is a shortcut: we first perform a (quicker) check to find the first non-identical object. Then switch to an equality check.

def __eq__(self, other):
    if len(xs) != len(ys):
        return False
    for x, y in zip(xs, ys, strict=True):
        if x is y:
            continue
        if x != y:
            return False
    return True

z = float('nan')
xs = [1, 2, 3, z]
ys = [1, 2, 3, z]

print(
    f'{xs == ys       = }',
    f'{__eq__(xs, ys) = }',
    sep='\n',
)
This is distinct from how numpy.ndarray equality works!

from numpy import array

z = float('nan')
xs = [1, 2, 3, z]
ys = [1, 2, 3, z]

assert xs == ys

z = float('nan')
xs = array([1, 2, 3, z])
ys = array([1, 2, 3, z])

assert not (xs == ys).all()
So why should I care…?

xs = ...
ys = ...

print(f'{xs is ys = }')