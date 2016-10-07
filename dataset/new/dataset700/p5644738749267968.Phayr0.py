#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import sys
import math

with open(sys.argv[1], 'r') as f:

    T = int(f.readline())
    for k in range(T):
        N = int(f.readline())
        Naomi = [float(val) for val in f.readline().split()]
        Naomi.sort()
        Ken = [float(val) for val in f.readline().split()]
        Ken.sort()
        oldKen = Ken
        fair = 0
        for i in range(N):
            j = i
            while (j < N and Naomi[i] > Ken[j]):
                j += 1
            if j == N :
                fair = N-i
                break
            else :
                temp = [Ken[j]] + Ken[i:j]
                if j+1< N:
                    temp += Ken[j+1:]
                Ken = Ken[:i] + temp
        deceitful=0
        lastN = N-1-i
        lastK = N-1-i
        firstN = 0
        firstK = 0
        for i in range(N):
            if oldKen[i] > Naomi[i]:
                temp = [oldKen[N-1]] + oldKen[i:N-1]
                oldKen = oldKen[:i] + temp
            else:
                deceitful +=1
        
        print(("Case #"+str(k+1)+": "+str(deceitful)+" "+str(fair)))
