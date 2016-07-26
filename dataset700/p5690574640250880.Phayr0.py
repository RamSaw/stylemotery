#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import sys
import math

with open(sys.argv[1], 'r') as f:

    line = f.readline()
    T = int(line)
    for k in range(T):
        R, C, M = [int(val) for val in f.readline().split()]
        rotate = False
        problem = False
        if R < C:
            temp = R
            R = C
            C = temp
            rotate = True
            
        res = []
        empty = R*C - M
        if C==1:
            res = [['.'] for i in range(R)]
            m = 0
            while m < M:
                res[m][0]='*'
                m += 1
            res[R-1][0] = 'c'
        elif empty >= 2*(R-2) + 2*C:
            res = [['.' for j in range(C)] for i in range(R)]
            m = 0
            while m < M:
                res[m/(C-2)][m%(C-2)]='*'
                m += 1
            res[R-1][C-1] = 'c'
            
        elif empty >= 2*C :
            if empty % C != 1:
                res = [['.' for j in range(C)] for i in range(R)]
                m = 0
                while m < M:
                    res[m/C][m%C]='*'
                    m += 1
                res[R-1][C-1] = 'c'
            elif empty >= 2*R:
                if empty % R != 1:
                    res = [['.' for j in range(C)] for i in range(R)]
                    m = 0
                    while m < M:
                        res[m%R][m/R]='*'
                        m += 1
                    res[R-1][C-1] = 'c'
                else:
                    problem = True
            else:
                problem = True
        elif empty==1:
            res = [['*' for j in range(C)] for i in range(R)]
            res[R-1][C-1]='c'
        elif empty >=4 and (empty %2) == 0:
            res = [['*' for j in range(C)] for i in range(R)]
            for i in range(empty/2):
                res[R-1-i][0] = '.'
                res[R-1-i][1] = '.'
            res[R-1][0] = 'c'   
        elif empty >=9 and (empty %2) ==1 :
            if C >= 3 and R >=4:
                res = [['*' for j in range(C)] for i in range(R)]
                for i in range(3):
                    res[R-1-i][0] = '.'
                    res[R-1-i][1] = '.'
                    res[R-1-i][2] = '.'
                res[R-1][0] = 'c' 
                for i in range((empty-9)/2):
                    res[R-4-i][0] = '.'
                    res[R-4-i][1] = '.'
            else:
                res=[]
        if problem and empty >=4 and (empty %2) == 0:
            res = [['*' for j in range(C)] for i in range(R)]
            if empty / 2 <= R:
                for i in range(empty/2):
                    res[R-1-i][0] = '.'
                    res[R-1-i][1] = '.'
                res[R-1][0] = 'c'
            else:
                res=[]
        elif problem and empty >=9 and (empty %2) ==1 :
            res = [['*' for j in range(C)] for i in range(R)]
            if C >= 3 and R >=4:
                for i in range(3):
                    res[R-1-i][0] = '.'
                    res[R-1-i][1] = '.'
                    res[R-1-i][2] = '.'
                res[R-1][0] = 'c' 
                for i in range((empty-9)/2):
                    res[R-4-i][0] = '.'
                    res[R-4-i][1] = '.'
            else:
                res=[]
            
        if len(res) != 0 and rotate:
            res2 = [['.' for j in range(R)] for i in range(C)]
            for i in range(C):
                line = ''
                for j in range(R):
                    res2[i][j] = res[j][i]
            res = res2 
        if len(res)==0:
            print(("Case #"+str(k+1)+":\nImpossible"))
        else:
            print(("Case #"+str(k+1)+":"))
            for i in range(len(res)):
                print((''.join(res[i])))
