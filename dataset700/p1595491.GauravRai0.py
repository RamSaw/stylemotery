fi=open("B-small-attempt0.in",'r')#Input File
#fi=open("B-large.in",'r')#Input File
#fi=open("B.in",'r')#Input File
fo=open("B-small-attempt0.out","w")#Output File
#fo=open("B-large.out","w")#Output File

T=int(fi.readline())
for case in range(1,T+1,1):
    temp = list(map(int, fi.readline().split()))
    n = temp[0]
    s = temp[1]
    p = temp[2]
    mx = 3 * p - 2
    mn = mx - 2 if mx > 1 else 11
    
    ans = 0
    s_cnt = 0
    for i in range(3, n+3):
        if temp[i] >= mx:
            ans += 1
        elif temp[i] >= mn:
            s_cnt += 1
    ans += min(s, s_cnt)        
        
    fo.write("Case #%s: %s\n"%(case, ans))
