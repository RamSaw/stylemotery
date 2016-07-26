fi=open("A-small-attempt0.in",'r')#Input File
fo=open("A-small-attempt0.out",'w')#Output File

#fi=open("A-large.in",'r')#Input File
#fo=open("A-large.out","w")#Output File

#fi = open("A.in",'r')#Input File
#fo = open("A.out","w")#Output File
    
T = int(fi.readline())
for case in range(1,T+1,1):
    ############################################
    ans = ""
    n = int(fi.readline())
    st1 = [set(map(int, fi.readline().split())) for i in range(4)]
    m = int(fi.readline())
    st2 = [set(map(int, fi.readline().split())) for i in range(4)]
    
    st = st1[n-1].intersection(st2[m-1])
    st_len = len(st)
    if st_len == 0:
        ans = "Volunteer cheated!"
    elif st_len == 1:
        ans = st.pop()
    else:
        ans = "Bad magician!"
    ############################################
    fo.write("Case #%s: %s\n"%(case, ans))
