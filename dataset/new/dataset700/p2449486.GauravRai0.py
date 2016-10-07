fi=open("B-small-attempt0.in",'r')#Input File
fo=open("B-small-attempt0.out",'w')#Output File

#fi=open("B-large.in",'r')#Input File
#fo=open("B-large.out","w")#Output File

#fi=open("B.in",'r')#Input File
#fo=open("B.out","w")#Output File


T=int(fi.readline())
for case in range(1,T+1,1):
	ans="YES"
	############################################
	n, m = list(map(int, fi.readline().split()))
	arr = [list(map(int, fi.readline().split())) for i in range(n)]
	org = [[100 for j in range(m)] for i in range(n)]
	lst = []
	for i in range(n):
	    lst.append((max(arr[i]), 0, i))
	for i in range(m):
	    mx = arr[0][i]
	    for j in range(1, n):
	        mx = max(mx, arr[j][i])
	    lst.append((mx, 1, i))
	lst.sort(reverse=True)
	for row in lst:
	    v, i, j = row
	    if i == 0:
	        for x in range(m):
	            org[j][x] = v
	    else:
	        for x in range(n):
	            org[x][j] = v
	if org != arr:
	    ans = "NO" 
	############################################
	fo.write("Case #%s: %s\n"%(case, ans))
