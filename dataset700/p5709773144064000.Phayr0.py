#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

import sys
import math

with open(sys.argv[1], 'r') as f:

    line = f.readline()
    T = int(line)
    for k in range(T):
        C, F, X = [float(val) for val in f.readline().split()]
        count = 0
        speed = 2.0
        while X/speed > C/speed + X/(speed + F):
            count += C / speed
            speed += F
        print(("Case #"+str(k+1)+": "+str(count + X/speed)))
