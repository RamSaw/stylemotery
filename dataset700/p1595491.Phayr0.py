#!/usr/bin/python

with open('./input.txt', 'r+') as f:
    with open('./output.txt', 'w') as fout:

        line = f.readline()
        T = int(line)
        for n in range(T):
            line = f.readline().split()
            N = int(line[0])
            S = int(line[1])
            p = int(line[2])
            res=0
            for i in range(N) :
                if (3*p-2<=int(line[3+i])):
                    res = res+1
                elif (S>0 and 3*p-4<=int(line[3+i]) and p<=int(line[3+i])):
                    res = res+1
                    S = S-1
            fout.write("Case #"+str(n+1)+": "+str(res)+"\n")
        
