def readFile(file):
    name = file[:file.index('.')]
    ##print(name)
    f = open(file)
    fout = open(name+'.out','w')
    cases = int(f.readline().strip())
    for case in range(cases):
        n,m = [int(x) for x in f.readline().split()]
        lawn = []
        for i in range(n):
            row = [int(x) for x in f.readline().split()]
            lawn.append(row)
        result = execute(case,lawn,n,m)
        print(result)
        fout.write(result)

def execute(index,lawn,n,m):
    print((index,lawn))

    result = ''
    for i in range(n):
        for j in range(m):
            h = lawn[i][j]
            row = True
            column = True
            #check row
            for x in range(m):
                if h < lawn[i][x]:
                    row = False
            #check column
            for x in range(n):
                if h < lawn[x][j]:
                    column = False

            if not(row or column):
                result = "NO"
                break
        if result:
            break

    if not result:
        result = "YES"
    
    return ''.join(['Case #',str(index+1),': ',str(result),'\n'])


if __name__ == "__main__":
    readFile('B-small-attempt0.in')
