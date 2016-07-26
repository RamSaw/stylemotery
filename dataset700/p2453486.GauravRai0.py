fi=open("A-small-attempt0.in",'r')#Input File
fo=open("A-small-attempt0.out",'w')#Output File

#fi=open("A-large.in",'r')#Input File
#fo=open("A-large.out","w")#Output File

#fi=open("A.in",'r')#Input File
#fo=open("A.out","w")#Output File


T=int(fi.readline())
for case in range(1,T+1,1):
    ans=0
    ############################################
    arr = [list(fi.readline().strip()) for i in range(4)]
    fi.readline()
    first = None
    won = False
    
    for i in range(4):
        first = arr[i][0] if arr[i][0] != 'T' else arr[i][1]
        if first == '.':
            continue
        won = True
        for j in range(4):
            if arr[i][j] != first and arr[i][j] != 'T':
                won = False
                break
        if won:
            break               
    if won:
        fo.write("Case #%s: %s\n"%(case, "%s won"%first))
        continue
                
    for i in range(4):
        first = arr[0][i] if arr[0][i] != 'T' else arr[1][i]
        if first == '.':
            continue
        won = True
        for j in range(4):
            if arr[j][i] != first and arr[j][i] != 'T':
                won = False
                break
        if won:
            break           
    if won:
        fo.write("Case #%s: %s\n"%(case, "%s won"%first))
        continue
        
    first = arr[0][0] if arr[0][0] != 'T' else arr[1][1]
    if first != '.':
        won = True
        for j in range(4):
            if arr[j][j] != first and arr[j][j] != 'T':
                won = False
                break    
        if won:
            fo.write("Case #%s: %s\n"%(case, "%s won"%first))
            continue
    
    first = arr[0][3] if arr[0][3] != 'T' else arr[1][2]
    if first != '.':
        won = True
        for j in range(4):
            if arr[j][3-j] != first and arr[j][3-j] != 'T':
                won = False
                break    
        if won:
            fo.write("Case #%s: %s\n"%(case, "%s won"%first))
            continue
    
    blank = False
    for i in range(4):
        for j in range(4):
            if arr[i][j] == '.':
                blank = True
                break    
    ############################################
    if blank:
        fo.write("Case #%s: Game has not completed\n"%(case))
    else:
        fo.write("Case #%s: Draw\n"%(case))    
