# -*- coding: utf-8 -*-

import math

def square_shelve(a, b):
    h = int(math.sqrt(a))
    if h ** 2 == a:
        mi = h
    else:
        mi = h + 1
    ma = int(math.sqrt(b))
    return [e ** 2 for e in range(mi, ma+1)]

def is_palindrome(n):
    if str(n) == str(n)[::-1]:
        return True
    return False

def solve(a, b):
    res = 0
    for e in square_shelve(a, b):
        if is_palindrome(e) and is_palindrome(int(math.sqrt(e))):
            res += 1
    return res

if __name__ == '__main__':
    t = int(input())
    for i in range(1, t+1):
        n, m = [int(e) for e in input().split(' ')]
        print('Case #%d:' % i, solve(n, m))

