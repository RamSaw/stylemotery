def readFile(file):
    name = file[:file.index('.')]
    ##print(name)
    f = open(file)
    fout = open(name+'.out','w')
    cases = int(f.readline().strip())
    for case in range(cases):
        a,b = [int(x) for x in f.readline().split()]
        
        result = execute(case,a,b)
        print(result)
        fout.write(result)

def execute(index,a,b):
    ##print(index,wires)
    count = 0
    #l = []

    for i in range(a,b+1):
        import math
        sq = math.sqrt(i)
        if sq.is_integer() and checkFair(i) and checkFair(int(sq)):            
            count += 1
            #l.append(i)

    #print(l)
        
    return ''.join(['Case #',str(index+1),': ',str(count),'\n'])

def checkFair(n):
    c = str(n)
    if len(c)==1:
        return True
    
    flag = True
    for i in range(len(c)//2):
        if c[i] != c[-(i+1)]:
            flag = False
            break
    return flag

def checkSquare(n):
    import math
    return math.sqrt(n).is_integer()

if __name__ == "__main__":
    readFile('C-small-attempt0.in')
