fi=open("C-small-attempt0.in",'r')#Input File
#fi=open("C-large.in",'r')#Input File
#fi=open("C.in",'r')#Input File
fo=open("C-small-attempt0.out","w")#Output File
#fo=open("C-large.out","w")#Output File

constant = 1001
mat = [0 for i in range(constant)]
flag = [False for i in range(constant)]
cur = 10
next = 100
no_of_rotations = 1
arr=[]

for i in range(11, constant):

    if flag[i]:
        continue
    flag[i] = True
               
    if i==next:
        cur = next
        next *= 10
        no_of_rotations += 1
        continue
        
    temp = [i]        
    num = i
    ind = 1
    
    for j in range(no_of_rotations):
         last_digit = num % 10
         num /= 10
         num += last_digit * cur
         
         if num <= i or num >= constant or flag[num]:
            continue
         
         flag[num] = True
         temp.append(num)
         ind += 1
         
    temp.sort()
    
    last = constant
    for j in range(0, ind-1):
        for k in range(j+1, ind):
            arr.append( [temp[j],temp[k]])

T=int(fi.readline())
for case in range(1,T+1,1):
    a, b = list(map(int, fi.readline().split()))
    ans = 0
    
    for ch in arr:
        if ch[0]>=a and ch[1]<=b:
            ans+=1
    print(case)
    fo.write("Case #%s: %s\n"%(case, ans))
