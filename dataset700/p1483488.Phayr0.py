#!/usr/bin/python
#import collections

pairs = set()
B = 2000001
for i in range(B) :
    for k in range(len(str(i))-1):
        m = int(str(i)[k+1:]+str(i)[0:k+1]) 
        if m<B and m>i :
            pairs.add((i,m))
    m = str(i)
    for j in range(len(str(B))-len(m)):
        m = m+'0' 
        if int(m)<B and int(m)>i :
            pairs.add((i,int(m)))
        

print(len(pairs))
with open('./input.txt', 'r+') as f:
    with open('./output.txt', 'w') as fout:

        line = f.readline()
        T = int(line)
        for k in range(T):
            A,B = [int(i) for i in f.readline().split()]
            res=0
            for (n,m) in pairs:
               if(n>=A and m<=B):
                   res = res+1
                
            fout.write("Case #"+str(k+1)+": "+str(res)+"\n")
        
