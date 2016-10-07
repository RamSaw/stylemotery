#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import sys
import math

with open(sys.argv[1], 'r') as f:

    line = f.readline()
    T = int(line)
    for k in range(T):
        l1 = int(f.readline())
        for j in range(l1):
            line = f.readline()
        liste1 = [int(w) for w in line.split()]
        for j in range(4-l1):
            line = f.readline()

        l2 = int(f.readline())
        for j in range(l2):
            line = f.readline()
        liste2 = [int(w) for w in line.split()]
        for j in range(4-l2):
            line = f.readline()
        
        collisions = []
        for i in liste1:
            for j in liste2:
                if i==j:
                    collisions.append(i)
        if len(collisions)==0:
            res = "Volunteer cheated!"
        elif len(collisions)>1:
            res = "Bad magician!"
        else:
            res = str(collisions[0])
        print(("Case #"+str(k+1)+": "+res))
