When would I actually write a class decorator… and is this really better than other approaches?
print("Let's take a look!")
A def-decorator performs the following syntactical transformation:

def dec(f): pass

@dec
def f(): pass

def f(): pass
f = dec(f)
Note that the common description of a decorator as a “function that takes a function and returns a function” is imprecise.

class T:
    def __init__(self, g):
        self.g = g

@T
def g():
    yield

print(f'{g = }')
A class-decorator performs the following syntactical transformation:

def dec(f): pass

@dec
class cls: pass

class cls: pass
cls = dec(cls)
Just as Python functions are defined and created at runtime, Python classes are also defined and created at runtime.

from random import Random
from inspect import signature

rnd = Random(0)
if rnd.choice([True, False]):
    class T:
        def f(self, x, y):
            return x * y
else:
    class T:
        def f(self, x):
            return x ** 2

print(f'{signature(T.f) = }')
Unlike the body of a function, for which bytecode is generated but not executed at function definite time, the body of a class is executed at class definition time.

from random import Random
from inspect import signature

rnd = Random(0)
class T:
    if rnd.choice([True, False]):
        def f(self, x, y):
            return x * y
    else:
        def f(self, x):
            return x ** 2

print(f'{signature(T.f) = }')
A Python class can have attributes added at runtime.

Unlike in Python 2, Python 3 doesn’t distinguish between bound and unbound methods. Instead, all Python functions support the __get__ descriptor protocol. The __get__ method is invoked when an attribute is looked up via the __getattr__/getattr protocol and is found on a class. When a function’s __get__ is invoked, it returns a method which binds the instance argument. Therefore, all Python 3 functions are unbound methods, and, therefore, it is relatively easy to add new methods to Python classes.

class T:
    pass

T.f = lambda self: ...

obj = T()
print(f'{obj.f() = }')
A class decorator receives the fully-constructed class and can therefore add, remove, or inspect attributes on that class. Note that a class decorator cannot distinguish the code that was statically written in the body of the class from code that was added to the class afterwards.

def dec(cls):
    print(f'{cls = }')
    return cls

@dec
class A:
    pass

@dec
class B(A):
    pass
Just as a def-decorator is used anytime we need to eliminate the risk of update anomaly associated with the definition of a function, a class decorator is about eliminating the risk of update anomaly associated with the definition of a class.

A class decorator could be used instead of inheritance to add functionality to a class without disrupting the inheritance hierarchy while potentially introducing modalities.

class A:
    def f(self):
        pass

class B(A):
    def g(self):
        pass

obj = B()
print(
    f'{obj.f() = }',
    f'{obj.g() = }',
    sep='\n',
)
def dec(cls):
    cls.f = lambda _: None
    return cls

@dec
class A:
    pass

@dec
class B(A):
    def g(self):
        pass

obj = B()
print(
    f'{obj.f() = }',
    f'{obj.g() = }',
    sep='\n',
)
def add_func(*funcs):
    def dec(cls):
        for name in funcs:
            setattr(cls, name, lambda _: None)
        return cls
    return dec

@add_func('f', 'g')
class A:
    pass

@add_func('f', 'h')
class B(A):
    def g(self):
        pass

obj = B()
print(
    f'{obj.f() = }',
    f'{obj.g() = }',
    f'{obj.h() = }',
    sep='\n',
)
A class-decorator can check that a class has certain contents (though it won’t be able to determine precisely how those contents were provided.)

def dec(cls):
    if not hasattr(cls, 'f'):
        raise TypeError('must define f')
    return cls

class A:
    def f(self):
        pass

@dec
class B(A):
    def f(self):
        pass
