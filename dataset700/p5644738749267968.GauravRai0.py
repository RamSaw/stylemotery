def play_optimally(arr1, arr2, n):
    ind = 0
    ans = 0
    for num in arr1:
        while ind < n and arr2[ind] < num:
            ind += 1
        if ind == n:
            break
        ans += 1
        ind += 1    
    return ans

fi=open("D-small-attempt0.in",'r')#Input File
fo=open("D-small-attempt0.out",'w')#Output File

#fi=open("D-large.in",'r')#Input File
#fo=open("D-large.out","w")#Output File

#fi = open("D.in",'r')#Input File
#fo = open("D.out","w")#Output File
    
T = int(fi.readline())
for case in range(1,T+1,1):
    ############################################
    n = int(fi.readline())
    arr1 = sorted(map(float, fi.readline().split()))
    arr2 = sorted(map(float, fi.readline().split()))
    
    ans1 = play_optimally(arr2, arr1, n)
    ans2 = n - play_optimally(arr1, arr2, n)

    ############################################
    fo.write("Case #%s: %s %s\n"%(case, ans1, ans2))
