#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import sys
import math

with open(sys.argv[1], 'r') as f:

    line = f.readline()
    T = int(line)
    for k in range(T):
        Game = []
        for i in range(4):
            line = f.readline()
            Game.append(line.strip('\n'))

        res = "Draw"
        empty_square = False
        for i in range(4):
            c = Game[i][0]
            if c == 'T':
                c = Game[i][1]
            if c == '.':
                empty_square = True
                continue
            over = True
            for j in range(1,4):
                over = over and (c == Game[i][j] or Game[i][j] == 'T')
                if Game[i][j] == '.':
                    empty_square = True
            if over:
                res = c
                break
        if res != "Draw":
            print(("Case #"+str(k+1)+": "+res+" won"))
            line = f.readline()
            continue
        
        for j in range(4):
            c = Game[0][j]
            if c == 'T':
                c = Game[1][j]
            if c == '.':
                continue
            over = True
            for i in range(1,4):
                over = over and (c == Game[i][j] or Game[i][j] == 'T')
            if over:
                res = c
                break
        if res != "Draw":
            print(("Case #"+str(k+1)+": "+res+" won"))
            line = f.readline()
            continue


        c = Game[0][0]
        if c != '.':
            if c == 'T':
                c = Game[1][1]
            if not c == '.':
                over = True
                for i in range(1,4):
                    over = over and (c == Game[i][i] or Game[i][i] == 'T')
                if over:
                    res = c
        if res != "Draw":
            print(("Case #"+str(k+1)+": "+res+" won"))
            line = f.readline()
            continue


        c = Game[0][3]
        if c != '.':
            if c == 'T':
                c = Game[1][2]
            if not c == '.':
                over = True
                for i in range(1,4):
                    over = over  and (c == Game[i][3-i] or Game[i][3-i] == 'T')
            if over:
                res = c
        if res != "Draw":
            print(("Case #"+str(k+1)+": "+res+" won"))
            line = f.readline()
            continue



        if empty_square :
            print(("Case #"+str(k+1)+": Game has not completed"))
        else :
            print(("Case #"+str(k+1)+": Draw"))
          
        line = f.readline()
