When would I actually write a metaclass… and is there a better way?
print("Let's take a look!")
A Python class allows us to implement the Python “vocabulary” by writing special __-methods.

class T:
    def __getitem__(self, key):
        pass
    def __len__(self):
        return 0

obj = T()
print(
    f'{obj[...] = }',
    f'{len(obj) = }',
    sep='\n',
)
These special __-methods are not looked up via the __getattr__ protocol. In CPython, they are looked up by direct C-struct access on type(…).

If we wanted to implement the Python vocabulary on a class object, we would need to implement these methods on whatever type(cls) is. This entity is called the “metaclass.”

A Python class is responsible for constructing its instances. A Python metaclass is responsible for constructing its instances, which happen to be Python classes.

from logging import getLogger, basicConfig, INFO

logger = getLogger(__name__)
basicConfig(level=INFO)

class TMeta(type):
    def __getitem__(self, key):
        logger.info('TMeta.__getitem__(%r, %r)', self, key)
        pass
    def __len__(self):
        logger.info('TMeta.__len__(%r)', self)
        return 0

class T(metaclass=TMeta):
    def __getitem__(self, key):
        logger.info('T.__getitem__(%r, %r)', self, key)
        pass
    def __len__(self):
        logger.info('T.__len__(%r)', self)
        return 0

obj = T()

obj[...]
len(obj)

T[...]
len(T)
This is not altogether that useful, in practice.

from logging import getLogger, basicConfig, INFO

logger = getLogger(__name__)
basicConfig(level=INFO)

class TMeta(type):
    def __call__(self, *args, **kwargs):
        obj = self.__new__(self, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        obj.__post_init__()
        return obj

class T(metaclass=TMeta):
    def __new__(cls, value):
        return super().__new__(cls)
    def __init__(self, value):
        self.value = value
    def __post_init__(self):
        self.value = abs(self.value)
    def __repr__(self):
        return f'T({self.value!r})'

obj = T(-123)
print(f'{obj = }')
Metaclasses are inherited down the class hierarchy. This is why, historically, they were used for enforcing constraints from base types to derived types.

Consider that Derived needs to constrain Base in order to operate correctly. However, this can be done trivially in app.py without touching any code in library.py.

from inspect import signature

# library.py
class Base:
    def helper(self):
        ...

# app.py
print(
    f'{signature(Base.helper) = }',
)
class Derived(Base):
    def func(self):
        return self.helper()
But if Base needs to constrain Derived, then this cannot be done so easily without putting code in app.py. Instead, we need to find some mechanism that operates at a higher level.

# library.py
class Base:
    def func(self):
        return self.implementation()

# app.py
class Derived(Base):
    def implementation(self):
        ...
The highest level mechanism we can employ to add a hook into the class construction process is builtins.__build_class__.

from functools import wraps
import builtins

@lambda f: setattr(builtins, f.__name__, f(getattr(builtins, f.__name__)))
def __build_class__(orig):
    @wraps(orig)
    def inner(func, name, *bases, **kwargs):
        print(f'{func, name, bases, kwargs = }')
        return orig(func, name, *bases)
        # return orig(func, name, *bases, **kwargs)
    return inner

class Base: pass
class Derived(Base): pass
class MoreDerived(Base, x=...): pass
What is the function that is passed to __build_class__?

from functools import wraps
import builtins

@lambda f: setattr(builtins, f.__name__, f(getattr(builtins, f.__name__)))
def __build_class__(orig):
    @wraps(orig)
    def inner(func, name, *bases, **kwargs):
        print(f'{func, name, bases, kwargs = }')
        print(f'{func() = }')
        # exec(func.__code__, globals(), ns := {})
        # print(f'{ns = }')
        return orig(func, name, *bases, **kwargs)
    return inner

class T:
    def f(self):
        pass
There’s not much we can do with __build_class__ other than debugging or instrumentation.

from functools import wraps
import builtins

@lambda f: setattr(builtins, f.__name__, f(getattr(builtins, f.__name__)))
def __build_class__(orig):
    @wraps(orig)
    def inner(func, name, *bases, **kwargs):
        print(f'{func, name, bases, kwargs = }')
        return orig(func, name, *bases, **kwargs)
    return inner

import json
# import pandas, matplotlib
Since a metaclass is inherited down the class hierarchy, it gives us a narrower hook-point. Additionally, the metaclass gets the partially constructed class, which is, in practice, more useful to work with.

class BaseMeta(type):
    def __new__(cls, name, bases, body, **kwargs):
        print(f'{cls, name, bases, body, kwargs = }')
        # return super().__new__(cls, name, bases, body, **kwargs)
        return super().__new__(cls, name, bases, body)

class Base(metaclass=BaseMeta):
    pass

class Derived(Base, x=...):
    pass
We can use this to enforce constraints.

# library.py
from inspect import signature

class BaseMeta(type):
    def __new__(cls, name, bases, body, **kwargs):
        rv = super().__new__(cls, name, bases, body, **kwargs)
        if rv.__mro__[-2::-1].index(rv):
            rv.check()
        return rv

class Base(metaclass=BaseMeta):
    @classmethod
    def check(cls):
        if not hasattr(cls, 'implementation'):
            raise TypeError('must implement method')
        if 'x' not in signature(cls.implementation).parameters:
            raise TypeError('method must take parameter named x')
    def func(self):
        return self.implementation()

# app.py
class Derived(Base):
    def implementation(self, x):
        ...
However, metaclasses tend to be tricky to write correctly, especially if you need to compose them.

# library.py
from inspect import signature

class BaseMeta(type):
    pass

class Base(metaclass=BaseMeta):
    pass

# app.py
class DerivedMeta(type):
    pass

class Derived(Base, metaclass=DerivedMeta):
    pass
# library.py
from inspect import signature

class Base0Meta(type):
    def __new__(cls, name, bases, body, **kwargs):
        print(f'Base0Meta.__new__({cls!r}, {name!r}, {bases!r}, {body!r}, **{kwargs!r})')
        return super().__new__(cls, name, bases, body, **kwargs)

class Base0(metaclass=Base0Meta):
    pass

class Base1Meta(type):
    def __new__(cls, name, bases, body, **kwargs):
        print(f'Base1Meta.__new__({cls!r}, {name!r}, {bases!r}, {body!r}, **{kwargs!r})')
        return super().__new__(cls, name, bases, body, **kwargs)

class Base1(metaclass=Base0Meta):
    pass

# app.py
class Derived(Base0, Base1):
    pass

class Derived(Base1, Base0):
    pass
# library.py
from inspect import signature

class Base0Meta(type):
    def __new__(cls, name, bases, body, **kwargs):
        print(f'Base0Meta.__new__({cls!r}, {name!r}, {bases!r}, {body!r}, **{kwargs!r})')
        return super().__new__(cls, name, bases, body, **kwargs)

class Base0(metaclass=Base0Meta):
    pass

class Base1Meta(type):
    def __new__(cls, name, bases, body, **kwargs):
        print(f'Base1Meta.__new__({cls!r}, {name!r}, {bases!r}, {body!r}, **{kwargs!r})')
        return super().__new__(cls, name, bases, body, **kwargs)

class Base1(metaclass=Base0Meta):
    pass

# app.py
class Derived(Base0):
    pass

class MoreDerived(Base1, Derived):
    pass
In Python 3.6, the __init_subclass__ mechanism was introduced. Like a metaclass, it is inherited down the class hierarchy. Unlike the metaclass, it gets the fully constructed class. __init_subclass__ doesn’t have the same compositional difficulties that metaclasses have.

class Base:
    def __init_subclass__(cls, **kwargs):
        print(f'{cls, kwargs = }')

class Derived(Base, x=...):
    pass
# library.py
from inspect import signature

class Base:
    def __init_subclass__(cls):
        if not hasattr(cls, 'implementation'):
            raise TypeError('must implement method')
        if 'x' not in signature(cls.implementation).parameters:
            raise TypeError('method must take parameter named x')
    def func(self):
        return self.implementation()

# app.py
class Derived(Base):
    def implementation(self, x):
        ...
class Base0:
    def __init_subclass__(cls):
        print(f'Base0.__init_subclass__({cls!r})')
        super().__init_subclass__()

class Base1:
    def __init_subclass__(cls):
        print(f'Base1.__init_subclass__({cls!r})')
        super().__init_subclass__()

class Derived0(Base0, Base1):
    pass

class Derived1(Base1, Base0):
    pass
class Base0:
    def __init_subclass__(cls):
        print(f'Base0.__init_subclass__({cls!r})')
        super().__init_subclass__()

class Base1:
    def __init_subclass__(cls):
        print(f'Base1.__init_subclass__({cls!r})')
        super().__init_subclass__()

class Derived(Base0):
    pass
# print(f'{Derived.__mro__ = }')

class MoreDerived0(Derived, Base1):
    pass
# print(f'{MoreDerived0.__mro__ = }')

class MoreDerived1(Base1, Derived):
    pass
# print(f'{MoreDerived1.__mro__ = }')
However, an __init_subclass__ requires that we interact with the inheritance hierarchy. But with a class-decorator, we do not. In the case of a class-decorator, we also get the fully-constructed class, but we don’t get any keyword arguments.

class Base:
    def __init_subclass__(cls, **kwargs):
        print(f'Base.__init_subclass__({cls!r}, **{kwargs!r})')

class Derived(Base, x=...):
    pass

def dec(cls):
    print(f'dec({cls!r})')
    return cls

@dec
class T:
    pass
However, we can write a higher-order class-decorator to introduce modalities.

class Base:
    def __init_subclass__(cls, **kwargs):
        print(f'Base.__init_subclass__({cls!r}, **{kwargs!r})')

class Derived(Base, x=...):
    pass

def d(**kwargs):
    def dec(cls):
        print(f'dec({cls!r}, **{kwargs!r})')
        return cls
    return dec

@d(x=...)
class T:
    pass
