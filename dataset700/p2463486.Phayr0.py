#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import sys
import math

#recognize palindromes
def is_palindrome(s):
    n = len(s)
    for i in range(n/2):
        if (s[i] != s[n-1-i]):
            return False
    return True

#precompute fair and square numbers
fair_and_squares = [1, 4, 9, 121, 484]
def precompute_fair_and_squares():
    fillings_k_2 = ['0', '1', '2']
    fillings_k_1 = ['00', '11', '22']
    for k in range(3, 50):
        fillings_k = []    
        for m in fillings_k_2:
            fillings_k.append('0'+m+'0')
            for s in ['1', '2']:
                n = int(s+m+s)
                if is_palindrome(str(n*n)):
                    fair_and_squares.append(n*n)
                    fillings_k.append(str(n))
        fillings_k_2 = fillings_k_1
        fillings_k_1 = fillings_k

#dichotomic research
def indices(A, B) :
    inf = 0
    sup = len(fair_and_squares)-1

    while (sup - inf) > 1:        
        mid = (sup + inf) /2
        if fair_and_squares[mid] <= A :
            inf = mid
        elif fair_and_squares[mid] > A:
            sup = mid
    if (fair_and_squares[inf] == A) :
        i_min = inf
    else:
        i_min = sup

    inf = 0
    sup = len(fair_and_squares)-1

    while (sup - inf) > 1:        
        mid = (sup + inf) /2
        if fair_and_squares[mid] <= B :
            inf = mid
        elif fair_and_squares[mid] > B:
            sup = mid
    if (fair_and_squares[sup] == B) :
        i_max = sup
    else:
        i_max = inf

    return i_min, i_max



with open(sys.argv[1], 'r') as f:
    precompute_fair_and_squares()
    fair_and_squares.sort()
    line = f.readline()
    T = int(line)
    for k in range(T):
        line = f.readline()
        A, B = [int(val) for val in line.split()]
        i_min, i_max = indices(A,B)
        print(("Case #"+str(k+1)+": "+str(i_max-i_min+1)))
          
