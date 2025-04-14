What is the difference between an object and a closure or an object and a generator coroutine… and how does this affect usability?
print("Let's take a look!")
In Python, we have first-class functions: functions can be treated like any other data. For example, we can put functions into data structures.

def f(x, y):
    return x + y

def g(x, y):
    return x * y

for f in [f, g]:
    print(f'{f(123, 456) = :,}')
    break

for rv in [f(123, 456), g(123, 456)]:
    print(f'{rv = :,}')
    break
We can also dynamically define new functions at runtime.

def f():
    def g(x):
        return x ** 2
    return g

g = f()
print(f'{g(123) = :,}')
Often, we may use lambda syntax if those functions are short (consisting of a single expression with no use of the ‘statement grammar.’)

for f in [lambda x, y: x + y, lambda x, y: x * y]:
    print(f'{f(123, 456) = :,}')
def f():
    return lambda x: x ** 2

g = f()
print(f'{g(123) = :,}')
We know that these functions are being defined dynamically, because every definition creates a new, distinct version of that function.

def f():
    def g(x):
        return x ** 2
    return g

g0, g1 = f(), f()

print(
    f'{g0(123)      = :,}',
    f'{g1(123)      = :,}',
    f'{g0 is not g1 = }',
    sep='\n',
)
Note that, in Python, we cannot compare functions for equality.

def f(x, y):
    return x + y
def g(x, y):
    return x + y

print(f'{f == g = }')
print(f'{f.__name__ == g.__name__ = }')
print(f'{f.__code__.co_code == g.__code__.co_code = }')
# funcs = {*()}
# for _ in range(3):
#     def f(x, y):
#         return x + y
#     funcs.add(f)

# for _ in range(3):
#     def f(x, y):
#         return x + y

print(f'{funcs = }')
When we dynamically define functions in Python, a function object is created that consists of the function’s name (whether anonymous or not,) its docstring (if provided,) its default values, its code object, and any non-local, non-global data it needs to operate (its closure.)

def f(x, ys=[123, 456]):
    '''
        adds x to each value in ys
    '''
    return [x + y for y in ys]

from dis import dis
dis(f)
print(
    # f'{f.__name__         = }',
    # f'{f.__doc__          = }',
    # f'{f.__code__         = }',
    # f'{f.__code__.co_code = }',
    # f'{f.__defaults__     = }',
    # f'{f.__closure__      = }',
    sep='\n',
)
Note that the defaults are created when the function is defined; this is why when we have “mutable default arguments,” there is only one copy of these defaults that is reüsed across all invocations of the function.

def f(xs=[123, 456]):
    xs.append(len(xs) + 1)
    return xs

print(
    f'{f()            = }',
    f'{f()            = }',
    f'{f.__defaults__ = }',
    f'{f() is f()     = }',
    sep='\n',
)
When the bytecode for a function is created, the Python compiler performs scope-determination. In order to generate the bytecodes for local variable access (LOAD_FAST,) for global variable access (LOAD_GLOBAL,) or for closure variable access (LOAD_DEREF,) the Python parser statically determines the scope of any variables that are used.

from dis import dis

def f():
    return x
# dis(f)

def f(x):
    # import x
    return x
dis(f)

def f(x):
    def g():
        nonlocal x
        x += 1
        return x
    return g
dis(f(...))
For variables that are neither local nor global but instead in the “enclosing environment,” we generate a LOAD_DEREF bytecode for access and capture a reference to that variable.

def f(x):
    def g():
        return x
    return g

xs = [1, 2, 3]
g = f(xs)

print(
    f'{g.__closure__                  = }',
    f'{g.__closure__[0]               = }',
    f'{g.__closure__[0].cell_contents = }',
    f'{g.__closure__[0].cell_contents is xs = }',
    sep='\n',
)
It is not a coïncidence that this is reminiscent of object orientation in Python. Just as an object “encapsulates” some (hidden) state and some behaviour that operates on such state, a dynamically defined function “closes over” some state that it can operate on.

class T:
    def __init__(self, state):
        self.state = state
    def __call__(self):
        self.state += 1
        return self.state
    def __repr__(self):
        return f'T({self.state!r})'

obj = T(123)
print(
    # f'{obj   = }',
    f'{obj() = }',
    f'{obj() = }',
    f'{obj() = }',
    sep='\n',
)
def create_obj(state):
    def f():
        nonlocal state
        state += 1
        return state
    return f

obj = create_obj(123)
print(
    f'{obj   = }',
    f'{obj() = }',
    f'{obj() = }',
    f'{obj() = }',
    sep='\n',
)
In fact, we can see the correspondence quite clearly when we look at what sits underneath.

class T:
    def __init__(self, state):
        self.state = state
    def __call__(self):
        self.state += 1
        return self.state
    def __repr__(self):
        return f'T({self.state!r})'

def create_obj(state):
    def f():
        nonlocal state
        state += 1
        return state
    return f

obj0 = T(123)
obj1 = create_obj(123)

print(
    f'{obj0.__dict__                     = }',
    f'{obj1.__closure__                  = }',
    f"{obj0.__dict__['state']            = }",
    f'{obj1.__closure__[0].cell_contents = }',
    sep='\n',
)
This tells us that an object created with the class keyword and a dynamically defined function created with a closure are two ways to accomplish the same goal of encapsulation.

When we create an instance a generator coroutine, it maintains its local state in-between iterations.

def coro(state):
    while True:
        state += 1
        yield state

ci = coro(123)
print(
    f'{next(ci) = }',
    f'{next(ci) = }',
    f'{next(ci) = }',
    f'{next(ci) = }',
    f'{ci.gi_frame.f_locals = }',
    sep='\n',
)
Indeed, this appears to be yet another way to accomplish the same goal!

class T:
    def __init__(self, state):
        self.state = state
    def __call__(self):
        self.state += 1
        return self.state
    def __repr__(self):
        return f'T({self.state!r})'

def f(state):
    def g():
        nonlocal state
        state += 1
        return state
    return g

def coro(state):
    while True:
        state += 1
        yield state

obj0 = T(123)
obj1 = f(123)
obj2 = coro(123).__next__

print(
    # f'{obj0() = } {obj0() = } {obj0() = }',
    # f'{obj1() = } {obj1() = } {obj1() = }',
    # f'{obj2() = } {obj2() = } {obj2() = }',

    # f'{obj0.__dict__                            = }',
    # f'{obj1.__closure__                         = }',
    # f'{obj2.__self__.gi_frame.f_locals          = }',
    f"{obj0.__dict__['state']                   = }",
    f'{obj1.__closure__[0].cell_contents        = }',
    f"{obj2.__self__.gi_frame.f_locals['state'] = }",
    sep='\n',
)
Facing three ways to accomplish the same goal, which do we choose?

choose class
If it makes sense for someone to be able to dig around into the internal details of the object, then maybe we should choose class.

class T:
    def __init__(self, state):
        self.state = state
    def __call__(self):
        self.state += 1
        return self.state
    def __repr__(self):
        return f'T({self.state!r})'
    def __dir__(self):
        return ['state']

obj = T(123)
print(
    f'{obj      = }',
    f'{dir(obj) = }',
    sep='\n',
)

def f(state):
    def g():
        nonlocal state
        state += 1
        return state
    return g

obj = f(123)
print(
    f'{obj      = }',
    f'{dir(obj) = }',
    f'{obj.__closure__ = }',
    sep='\n',
)
If it makes sense for the object to support multiple named methods, then class is probably less clumsy.

class T:
    def __init__(self, state):
        self.state = state
    def inc(self):
        self.state += 1
        return self.state
    def dec(self):
        self.state -= 1
        return self.state
    def __repr__(self):
        return f'T({self.state!r})'

obj = T(123)

print(
    f'{dir(obj) = }',
    # f'{obj.inc() = }',
    # f'{obj.dec() = }',
    sep='\n',
)
from collections import namedtuple

def f(state):
    def inc():
        nonlocal state
        state += 1
        return state
    def dec():
        nonlocal state
        state -= 1
        return state
    # return inc, dec
    return namedtuple('T', 'inc dec')(inc, dec)

# obj = f(123)
# print(
#     # f'{dir(obj) = }',
#     f'{obj[0]() = }',
#     f'{obj[1]() = }',
#     sep='\n',
# )

inc, dec = f(123)
print(
    f'{inc() = }',
    f'{dec() = }',
    sep='\n',
)

# obj = f(123)
# print(
#     f'{obj.inc() = }',
#     f'{obj.dec() = }',
#     sep='\n',
# )
If we need to implement any other parts of the Python vocabulary, then we must write class (or use some boilerplate elimination tool like contextlib.contextmanager.)

class T:
    def __init__(self, state):
        self.state = state
    def __call__(self, value):
        self.state.append(value)
    def __len__(self):
        return len(self.state)
    def __getitem__(self, idx):
        return self.state[idx]
    def __repr__(self):
        return f'T({self.state!r})'

obj = T([1, 2, 3])
obj(4)

print(
    f'{len(obj) = }',
    f'{obj[0]   = }',
    sep='\n',
)
choose closure
If we want to “hide” data from our users to limit them in some antagonistic or coërcive way, we should not expect the closure to add anything but few easily circumventable steps.

def f(state):
    def g():
        nonlocal state
        state += 1
        return state
    return g

g = f(123)
g.__closure__[0].cell_contents = 456

print(
    f'{g() = }',
    f'{g.__closure__[0].cell_contents = }',
    sep='\n',
)
This is not too dissimilar from our guidance around @property.

class T:
    def __init__(self, x):
        self._x = x

    @property
    def x(self):
        return self._x

    def __repr__(self):
        return f'T({self._x})'

obj = T(123)
# obj.x = ...
obj._x = ...
No matter how deeply we try to hide some data, it’s only a few dirs away.

def f(x):
    class T:
        @property
        def x(self):
            return x

        def __repr__(self):
            return f'T({self._x})'
    return T()

obj = f(123)

print(
    f'{obj.x = }',
    f'{type(obj).x.fget.__closure__[0].cell_contents = }',
    sep='\n',
)
If we want to non-antagonistically reduce clutter or noise, we may choose to use a closure.

class T:
    def __init__(self, state):
        self.state = state
    def __call__(self):
        self.state += 1
        return self.state
    def __repr__(self):
        return f'T({self.state!r})'

def f(state):
    def g():
        nonlocal state
        state += 1
        return state
    return g

obj0 = T(123)
obj1 = f(123)

print(
    f'{obj0   = }',
    f'{obj1   = }',
    f'{obj0() = }',
    f'{obj1() = }',
    sep='\n',
)
choose generator coroutine
If we have a heterogeneous computation, we generally do not want a generator coroutine if the computation will be triggered manually.

from dataclasses import dataclass

@dataclass
class State:
    a : int = None
    b : int = None
    c : int = None

class T:
    def __init__(self, state : State = None):
        self.state = state if state is not None else State()
    def f(self, value):
        self.state.a = value
        return self.state
    def g(self, value):
        self.state.b = self.state.a + value
        return self.state
    def h(self, value):
        self.state.c = self.state.b + value
        return self.state

obj = T()
print(
    f'{obj.f(123) = }',
    f'{obj.g(456) = }',
    f'{obj.h(789) = }',
    sep='\n',
)
from dataclasses import dataclass

@dataclass
class State:
    a : int = None
    b : int = None
    c : int = None

def coro(state : State = None):
    state = state if state is not None else State()
    state.a = yield ...
    state.b = (yield state) + state.a
    state.c = (yield state) + state.b
    yield state

obj = coro(); next(obj)
print(
    f'{obj.send(123) = }',
    ...
    f'{obj.send(456) = }', # ???
    ...
    ...
    ...
    f'{obj.send(789) = }',
    sep='\n',
)
If we have a single, homogeneous decomposition of a computation, we may find a generator coroutine is less conceptual overhead than a class-style object.

def coro():
    while True:
        _ = yield

ci = coro()
print(
    # f'{dir(ci) = }',
    f'{next(ci)              = }',
    f'{ci.send(...)          = }',
    # f'{ci.throw(Exception()) = }',
    # f'{ci.close()            = }',
    sep='\n',
)
In fact, we may find that pumped generator coroutines with __call__-interface unification give us an extremely simple API we can present our users.

from functools import wraps

def f(x):
    pass

def g():
    def f(x):
        pass
    return f

class T:
    def __call__(self, x):
        pass

@lambda coro: wraps(coro)(lambda *a, **kw: [ci := coro(*a, **kw), next(ci), ci.send][-1])
def coro():
    while True:
        _ = yield
How does this affect usability