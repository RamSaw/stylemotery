import os, sys
import ast
#from os import system


class foo:
    def __init__(self, i):
        print(100)


def func1(arg1, arg2):
    arg3 = arg1 + arg2
    return arg3 * 3.0


def func2(arg1, arg2):
    arg3 = arg1 * arg2
    return arg3 ** 2


i = 10 * 10 * 10 * 10
j = 2
me = "just dumpy str with no purpose"
bander = foo(22)

if i == j:
    print("EQUAL")
else:
    print("NOT EQUAL")

print(bander.__dict__)
print(func1(i, j))
print(func2(i, j))
