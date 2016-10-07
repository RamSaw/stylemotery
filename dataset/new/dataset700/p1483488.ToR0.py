def readFile(file):
    name = file[:file.index('.')]
    ##print(name)
    f = open(file)
    fout = open(name+'.out','w')
    cases = int(f.readline().strip())
    for i in range(cases):
        case = f.readline().strip().split()
        a = int(case[0])
        b = int(case[1])
        result = execute(i,a,b)
        print(result)
        fout.write(result)

def execute(index,a,b):
    print((index,a,b))
    count=0
    result=[]

    for x in range(a,b+1):
        numbers = generateRecycle(x)
        for y in numbers:
            if y > x and y <= b:
                result.append((x,y))
                count+=1

    testValue(result)

    #print(count)
    #print(result)
    
    return ''.join(['Case #',str(index+1),': ',str(count),'\n'])
        
def generateRecycle(x):
    result =[]
    strVal = str(x)
    if(len(strVal)>1):
        t = strVal
        for i in range(len(strVal)):
            t = ''.join((t[len(t)-1],t[:len(t)-1]))
            if int(t) != x :
                result.append(int(t))
        
    return set(result)
    
def testValue(result):
    for i1 in range(len(result)):
        for i2 in range(i1+1,len(result)):
            r1 = result[i1]
            r2 = result[i2]
            if(r1[0]==r2[1] and r1[1]==r2[0]) or (r1[0]==r2[0] and r1[1]==r2[1]):
                print(("error",r1,r2))

if __name__ == "__main__":
    readFile('C-small-attempt0.in')
