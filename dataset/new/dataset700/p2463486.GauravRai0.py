fi=open("C-small-attempt0.in",'r')#Input File
fo=open("C-small-attempt0.out",'w')#Output File

#fi=open("C-large.in",'r')#Input File
#fo=open("C-large.out","w")#Output File

#fi=open("C.in",'r')#Input File
#fo=open("C.out","w")#Output File

import math

def is_palindrom(n):
    x = str(n)
    y = x[::-1]
    return x == y
    
T=int(fi.readline())
for case in range(1,T+1,1):
	ans=0
	############################################
	a, b = list(map(int, fi.readline().split()))
	a = int(math.ceil(math.sqrt(a)))
	b = int(math.floor(math.sqrt(b)))
	for i in range(a, b+1):
	    if is_palindrom(i) and is_palindrom(i**2):
	        ans += 1
	        
	############################################
	fo.write("Case #%s: %s\n"%(case, ans))
