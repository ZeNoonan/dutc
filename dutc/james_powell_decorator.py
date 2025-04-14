When would I actually write a decorator or a higher-order decorator… and why?
print("Let's take a look!")
Python function definitions are executed at runtime.

def f():
    pass

print(f'{f = }')
This is why we can conditionally define functions or define funcitons in other functions. In Python, we can treat functions like any other data.

from random import Random
from inspect import signature

rnd = Random(0)

if rnd.choice([True, False]):
    def f(x, y):
        return x + y
else:
    def f(x):
        return -x

print(
    f'{f            = }',
    f'{signature(f) = }',
    sep='\n'
)
from types import FunctionType

def f(): pass

f = FunctionType(
    f.__code__,
    f.__globals__,
    name=f.__name__,
    argdefs=f.__defaults__,
    closure=f.__closure__,
)

print(
    f'{f              = }',
    f'{f.__code__     = }',
    f'{f.__globals__  = }',
    f'{f.__defaults__ = }',
    f'{f.__closure__  = }',
    sep='\n'
)
def f(x): return x + 1
def g(x): return x * 2
def h(x): return x ** 3

for func in [f, g, h]:
    print(f'{func(123) = :,}')

for rv in [f(123), g(123), h(123)]:
    print(f'{rv = :,}')

FUNCS = {
    'eff':  f,
    'gee':  g,
    'aich': h,
}
for name in 'eff eff gee aich'.split():
    print(f'{FUNCS[name](123) = :,}')
When we define a function in Python, it “closes” over its defining environment. In other words, if the function accesses data that is neither in the global scope nor local scope (but in the enclosing function’s scope,) we create a means to access this data. Note that this does not mean that we capture a reference to the data; the closure is its own indirection.

from dis import dis

def f(y):
    def g(z):
        return x + y + z
    return g

x = 1
g = f(y=20)

print(
    f'{g(z=300) = }',
    sep='\n',
)
# dis(g)
def f(y):
    def g(z):
        return x + y + z
    return g

x = 1
g = f(y=20)

print(
    f'{g.__closure__ = }',
    f'{g.__closure__[0].cell_contents = }',
    sep='\n',
)
def f(x):
    def g0():
        return x
    def g1():
        return x
    return g0, g1

g0, g1 = f(123)

print(
    f'{g0.__closure__ = }',
    f'{g1.__closure__ = }',
    f'{g0.__closure__[0].cell_contents = }',
    f'{g1.__closure__[0].cell_contents = }',
    sep='\n',
)
from math import prod

def f(xs):
    def g0():
        xs.append(sum(xs))
        return xs
    def g1():
        xs.append(prod(xs))
        return xs
    return g0, g1

g0, g1 = f([1, 2, 3])

print(
    f'{g0() = }',
    f'{g1() = }',
    f'{g0() = }',
    f'{g1() = }',
    sep='\n',
)
from math import prod

def f(x):
    def g0():
        nonlocal x
        x += 2
        return x
    def g1():
        nonlocal x
        x *= 2
        return x
    return g0, g1

g0, g1 = f(123)

print(
    f'{g0() = }',
    # f'{g0.__closure__[0], } · {g1.__closure__[0] = }',
    f'{g1() = }',
    # f'{g0.__closure__[0], } · {g1.__closure__[0] = }',
    f'{g0() = }',
    # f'{g0.__closure__[0], } · {g1.__closure__[0] = }',
    f'{g1() = }',
    # f'{g0.__closure__[0], } · {g1.__closure__[0] = }',
    sep='\n',
)
This will be important later.

Recall that functions are a means by which we can eliminate “update anomalies.” They represent a “single source of truth” for how to perform an operation.

We want to distinguish between “coïncidental” and “intentional” repetition. In the case of “intentional” repetition, we want to write a function; in the case of “coïncidental” repetition, we may not want to write a function.

# library.py
from random import Random
from statistics import mean, pstdev
from string import ascii_lowercase
from itertools import groupby

def generate_data(*, random_state=None):
    rnd = Random() if random_state is None else random_state
    return {
        ''.join(rnd.choices(ascii_lowercase, k=2)): rnd.randint(-100, +100)
        for _ in range(100)
    }

def normalise_data(data):
    μ,σ = mean(data.values()), pstdev(data.values())
    return {k: (v - μ) / σ for k, v in data.items()}

def process_data(data):
    return groupby(sorted(data.items(), key=(key := lambda k_v: k_v[0][0])), key=key)

def report(results):
    for k, g in results:
        g = dict(g)
        print(f'{k:<3} {min(g.values()):>5.2f} ~ {max(g.values()):>5.2f}')

# script0.py
if __name__ == '__main__':
    rnd = Random(0)
    raw_data = generate_data(random_state=rnd)
    data = normalise_data(raw_data)
    results = process_data(data)
    report(results)

# script1.py
if __name__ == '__main__':
    rnd = Random(0)
    raw_data = generate_data(random_state=rnd)
    data = normalise_data(raw_data)
    results = process_data(data)
    report(results)
def do_report():
    rnd = Random(0)
    raw_data = generate_data(random_state=rnd)
    data = normalise_data(raw_data)
    results = process_data(data)
    report(results)

# script0.py
if __name__ == '__main__':
    do_report()

# script1.py
if __name__ == '__main__':
    do_report()
def do_report(normalise=True):
    rnd = Random(0)
    raw_data = generate_data(random_state=rnd)
    if normalise:
        data = normalise_data(raw_data)
    results = process_data(data)
    report(results)

# script0.py
if __name__ == '__main__':
    do_report(normalise=False)

# script1.py
if __name__ == '__main__':
    do_report()
def report(results, prec=2):
    for k, g in results:
        g = dict(g)
        print(f'{k:<3} {min(g.values()):>{2+1+prec}.{prec}f} ~ {max(g.values()):>{2+1+prec}.{prec}f}')

def do_report(normalise=True, digits_prec=None):
    rnd = Random(0)
    raw_data = generate_data(random_state=rnd)
    if normalise:
        data = normalise_data(raw_data)
    results = process_data(data)
    if digits_prec is not None:
        report(results, prec=digits_prec)
    else:
        report(results)

# script0.py
if __name__ == '__main__':
    do_report(normalise=False)

# script1.py
if __name__ == '__main__':
    do_report(digits_prec=5)
If the functions provided by our analytical libraries represent the base-most, atomic units of our work, we could describe the common progression of effort as starting with manual composition of these units. Where patterns arise and intentional repetition is found, our primary work may move to managing this composition: writing classes and functions. Our work may continue to grow more abstract and we may discover patterns and intentional repetition across the writing of functions.

f()
g()
f()
h()
def func0():
    f()
    g()
    f()

def func1():
    f(g())

func0()
func1()
Mechanically, the @ syntax in Python is simple shorthand.

@dec
def f():
    pass

# … means…

def f():
    pass
f = dec(f)
This is key to understanding all of the mechanics behind decorators.

The simplest example of decorators is a system in which we need to instrument some code.

from random import Random
from time import sleep

def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __name__ == '__main__':
    print(f'{fast(123, 456) = :,}')
    print(f'{slow(123)      = :,}')
    print(f'{slow(456)      = :,}')
    print(f'{fast(456, 789) = :,}')
from random import Random
from time import sleep, perf_counter

def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __name__ == '__main__':
    before = perf_counter()
    print(f'{fast(123, 456) = :,}')
    after = perf_counter()
    print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
    before = perf_counter()
    print(f'{slow(123)      = :,}')
    after = perf_counter()
    print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
    before = perf_counter()
    print(f'{slow(456)      = :,}')
    print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
    before = perf_counter()
    print(f'{fast(456, 789) = :,}')
    after = perf_counter()
    print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
from random import Random
from time import sleep, perf_counter

def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __name__ == '__main__':
    if __debug__: before = perf_counter()
    print(f'{fast(123, 456) = :,}')
    if __debug__:
        after = perf_counter()
        print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
    if __debug__: before = perf_counter()
    print(f'{slow(123)      = :,}')
    if __debug__:
        after = perf_counter()
        print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
    if __debug__: before = perf_counter()
    print(f'{slow(456)      = :,}')
    if __debug__:
        after = perf_counter()
        print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
    if __debug__: before = perf_counter()
    print(f'{fast(456, 789) = :,}')
    if __debug__:
        after = perf_counter()
        print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
from random import Random
from time import sleep, perf_counter

def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __debug__:
    def bef():
        global before
        before = perf_counter()
    def aft():
        after = perf_counter()
        print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
else:
    def bef(): pass
    def aft(): pass

if __name__ == '__main__':
    bef()
    print(f'{fast(123, 456) = :,}')
    aft()
    bef()
    print(f'{slow(123)      = :,}')
    aft()
    bef()
    print(f'{slow(456)      = :,}')
    aft()
    bef()
    print(f'{fast(456, 789) = :,}')
    aft()
from random import Random
from time import sleep, perf_counter

def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __debug__:
    def bef():
        global before
        before = perf_counter()
    def aft():
        after = perf_counter()
        print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
else:
    def bef(): pass
    def aft(): pass

if __name__ == '__main__':
    bef()
    print(f'{fast(123, 456) = :,}')
    aft()
    bef()
    print(f'{slow(123)      = :,}')
    aft()
    bef()
    print(f'{slow(456)      = :,}')
    aft()
    bef()
    print(f'{fast(456, 789) = :,}')
    aft()
from random import Random
from time import sleep, perf_counter

def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __debug__:
    def timed(func, *args, **kwargs):
        before = perf_counter()
        rv = func(*args, **kwargs)
        after = perf_counter()
        print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
        return rv
else:
    def timed(func, *args, **kwargs):
        return func(*args, **kwargs)

if __name__ == '__main__':
    print(f'{timed(fast, 123, 456) = :,}')
    print(f'{timed(slow, 123)      = :,}')
    print(f'{timed(slow, 456)      = :,}')
    print(f'{timed(fast, 456, 789) = :,}')
from random import Random
from time import sleep, perf_counter

def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __debug__:
    def timed(func):
        def inner(*args, **kwargs):
            before = perf_counter()
            rv = func(*args, **kwargs)
            after = perf_counter()
            print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
            return rv
        return inner
else:
    def timed(func):
        return func

if __name__ == '__main__':
    print(f'{timed(fast)(123, 456) = :,}')
    print(f'{timed(slow)(123)      = :,}')
    print(f'{timed(slow)(456)      = :,}')
    print(f'{timed(fast)(456, 789) = :,}')
from random import Random
from time import sleep, perf_counter

def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __debug__:
    def timed(func):
        def inner(*args, **kwargs):
            before = perf_counter()
            rv = func(*args, **kwargs)
            after = perf_counter()
            print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
            return rv
        return inner
    fast, slow = timed(fast), timed(slow)

if __name__ == '__main__':
    print(f'{fast(123, 456) = :,}')
    print(f'{slow(123)      = :,}')
    print(f'{slow(456)      = :,}')
    print(f'{fast(456, 789) = :,}')
from random import Random
from time import sleep, perf_counter

def timed(func):
    def inner(*args, **kwargs):
        before = perf_counter()
        rv = func(*args, **kwargs)
        after = perf_counter()
        print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
        return rv
    return inner

def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y
if __debug__: fast = timed(fast)

def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2
if __debug__: slow = timed(slow)

if __name__ == '__main__':
    print(f'{fast(123, 456) = :,}')
    print(f'{slow(123)      = :,}')
    print(f'{slow(456)      = :,}')
    print(f'{fast(456, 789) = :,}')
from random import Random
from time import sleep, perf_counter

def timed(func):
    def inner(*args, **kwargs):
        before = perf_counter()
        rv = func(*args, **kwargs)
        after = perf_counter()
        print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
        return rv
    return inner

@timed if __debug__ else lambda f: f
def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

@timed if __debug__ else lambda f: f
def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __name__ == '__main__':
    print(f'{fast(123, 456) = :,}')
    print(f'{slow(123)      = :,}')
    print(f'{slow(456)      = :,}')
    print(f'{fast(456, 789) = :,}')
from random import Random
from time import sleep, perf_counter

def timed(func):
    def inner(*args, **kwargs):
        before = perf_counter()
        rv = func(*args, **kwargs)
        after = perf_counter()
        print(f'\N{mathematical bold capital delta}t: {after - before:.2f}s')
        return rv
    inner.orig = func
    return inner

@timed if __debug__ else lambda f: f
def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

@timed if __debug__ else lambda f: f
def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __name__ == '__main__':
    print(f'{fast(123, 456) = :,}')
    print(f'{slow(123)      = :,}')
    print(f'{slow.orig(456) = :,}')
    print(f'{fast(456, 789) = :,}')
    # help(fast)
from random import Random
from time import sleep, perf_counter
from functools import wraps, cached_property
from collections import deque, namedtuple
from datetime import datetime

class Call(namedtuple('CallBase', 'timestamp before after func args kwargs')):
    @cached_property
    def elapsed(self):
        return self.after - self.before

def timed(telemetry):
    def dec(func):
        @wraps(func)
        def inner(*args, **kwargs):
            before = perf_counter()
            rv = func(*args, **kwargs)
            after = perf_counter()
            telemetry.append(
                Call(datetime.now(), before, after, func, args, kwargs)
            )
            return rv
        inner.orig = func
        return inner
    return dec

telemetry = []

@timed(telemetry) if __debug__ else lambda f: f
def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

@timed(telemetry) if __debug__ else lambda f: f
def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __name__ == '__main__':
    print(f'{fast(123, 456) = :,}')
    print(f'{slow(123)      = :,}')
    print(f'{slow.orig(456) = :,}')
    print(f'{fast(456, 789) = :,}')

    for x in telemetry:
        print(f'{x.func.__name__} \N{mathematical bold capital delta}t: {x.elapsed:.2f}s')
from random import Random
from time import sleep, perf_counter
from functools import wraps, cached_property
from collections import deque, namedtuple
from datetime import datetime
from contextvars import ContextVar
from contextlib import contextmanager, nullcontext
from inspect import currentframe, getouterframes

def instrumented(func):
    if not __debug__:
        return func
    @wraps(func)
    def inner(*args, **kwargs):
        ctx = inner.context.get(nullcontext)
        frame = getouterframes(currentframe())[1]
        with ctx(frame, func, args, kwargs) if ctx is not nullcontext else ctx():
            return func(*args, **kwargs)
    @contextmanager
    def with_measurer(measurer):
        token = inner.context.set(measurer)
        try: yield
        finally: pass
        inner.context.reset(token)
    inner.with_measurer = with_measurer
    inner.context = ContextVar('context')
    return inner

@instrumented
def fast(x, y, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.1, .2))
    return x + y

@instrumented
def slow(x, *, random_state=None):
    rnd = Random() if random_state is None else random_state
    sleep(rnd.uniform(.25, .5))
    return x**2

if __name__ == '__main__':
    class Call(namedtuple('CallBase', 'lineno timestamp before after func args kwargs')):
        @cached_property
        def elapsed(self):
            return self.after - self.before
        def __str__(self):
            if self.args and self.kwargs:
                params = (
                    f'{", ".join(f"{x!r}" for x in self.args)}, '
                    f'{", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())}'
                )
            elif self.args:
                params = f'{", ".join(f"{x!r}" for x in self.args)}'
            elif self.kwargs:
                params = f'{", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())}'
            else:
                params = ''
            return f'{self.func.__name__}({params})'

        telemetry = []

        @classmethod
        @contextmanager
        def timed(cls, frame, func, args, kwargs):
            before = perf_counter()
            try: yield
            finally: pass
            after = perf_counter()
            cls.telemetry.append(
                cls(frame.lineno, datetime.now(), before, after, func, args, kwargs)
            )

    with fast.with_measurer(Call.timed), slow.with_measurer(Call.timed):
        print(f'{fast(123, 456) = :,}')
        print(f'{slow(123)      = :,}')
    print(f'{slow(456)      = :,}')
    print(f'{fast(456, 789) = :,}')

    for x in Call.telemetry:
        print(f'@line {x.lineno}: {x!s:<20} \N{mathematical bold capital delta}t {x.elapsed:.2f}s')
