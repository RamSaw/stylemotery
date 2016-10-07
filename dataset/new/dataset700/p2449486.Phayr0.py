#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import sys
import math

with open(sys.argv[1], 'r') as f:

    A = {}
    line = f.readline()
    T = int(line)
    for k in range(T):
        line = f.readline()
        N, M = [int(val) for val in line.split()]
        heights_l = []
        heights_c = []
        for i in range(N):
            line = f.readline()
            l = [int(val) for val in line.split()]
            heights_l.append(max(l))
            A[i] = l
        #constraints on height commands
        for j in range(M):
            maxi = A[0][j]
            for i in range(1,N):
                if A[i][j] > maxi:
                    maxi = A[i][j]
            heights_c.append(maxi)

        #check if those commands produce the right heights
        ok = True
        for i in range(N):
            for j in range(M):
                if A[i][j] != min(heights_l[i], heights_c[j]) :
                    ok = False
                    break
            if not ok:
                break

        if ok :
            print(("Case #"+str(k+1)+": YES"))
        else :
            print(("Case #"+str(k+1)+": NO"))
          
