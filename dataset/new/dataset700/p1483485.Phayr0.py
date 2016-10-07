#!/usr/bin/python

from string import maketrans   

googlerese = "qzejp mysljylc kd kxveddknmc re jsicpdrysi rbcpc ypc rtcsra dkh wyfrepkym veddknkmkrkcd de kr kd eoya kw aej tysr re ujdr lkgc jv"
english = "zqour language is impossible to understand there are twenty six factorial possibilities so it is okay if you want to just give up"

trad = maketrans(googlerese, english)

with open('./input.txt', 'r+') as f:
    with open('./output.txt', 'w') as fout:

        line = f.readline()
        T = int(line)
        for n in range(T):
            line = f.readline()
            fout.write("Case #"+str(n+1)+": "+line.translate(trad))
        
