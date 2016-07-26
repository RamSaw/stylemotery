fi=open("B-small-attempt0.in",'r')#Input File
fo=open("B-small-attempt0.out",'w')#Output File

#fi=open("B-large.in",'r')#Input File
#fo=open("B-large.out","w")#Output File

#fi=open("B.in",'r')#Input File
#fo=open("B.out","w")#Output File
    
T=int(fi.readline())
for case in range(1,T+1,1):
    ############################################
    c, f, x = list(map(float, fi.readline().split()))
    r = 2
    ans = 0
    if c >= x:
        ans = x / r
    else:
        while True:
            cur = x / r
            next = c / r
            next_total = next + (x / (r+f))
            if cur <= next_total:
                ans += cur
                break
            else:
                ans += next
                r += f
    ############################################
    fo.write("Case #%s: %.7f\n"%(case, ans))
