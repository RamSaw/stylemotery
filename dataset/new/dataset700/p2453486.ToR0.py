def readFile(file):
    name = file[:file.index('.')]
    ##print(name)
    f = open(file)
    fout = open(name+'.out','w')
    cases = int(f.readline().strip())
    for case in range(cases):
        board = []
        for i in range(4):
            l = list(f.readline().strip())
            board.append(l)
        f.readline()
            
        result = execute(case,board)
        print(result)
        fout.write(result)

def execute(index,board):
    print((index,board))

    #fixed
    k = 4
    g = replaceTO(board)
    h = replaceTX(board)

    print(g)
    print(h)

    o = False
    x = False
    space = False
    for i in range(k):
        for j in range(k):
            c = g[i][j]
            if c != '.':
                #check right                
                if checkright(g,i,j,k):
                    if c == "O" :
                        o = True
                    elif c == "X" :
                        x = True                        
                    
                #check down
                if checkdown(g,i,j,k):
                    if c == "O" :
                        o = True
                    elif c == "X" :
                        x = True

                #check downright
                if checkdownright(g,i,j,k):
                    if c == "O" :
                        o = True
                    elif c == "X" :
                        x = True

                #check downleft
                if checkdownleft(g,i,j,k):
                    if c == "O" :
                        o = True
                    elif c == "X" :
                        x = True
            else:
                space = True

    for i in range(k):
        for j in range(k):
            c = h[i][j]
            if c != '.':
                #check right                
                if checkright(h,i,j,k):
                    if c == "O" :
                        o = True
                    elif c == "X" :
                        x = True
                    
                #check down
                if checkdown(h,i,j,k):
                    if c == "O" :
                        o = True
                    elif c == "X" :
                        x = True

                #check downright
                if checkdownright(h,i,j,k):
                    if c == "O" :
                        o = True
                    elif c == "X" :
                        x = True

                #check downleft
                if checkdownleft(h,i,j,k):
                    if c == "O" :
                        o = True
                    elif c == "X" :
                        x = True
            else:
                space = True

    result = ''
    if o and x:
        result = "Both"
    elif o and not x:
        result = "O won"
    elif not o and x:
        result = "X won"
    else:
        if space :
            result = "Game has not completed"
        else:
            result = "Draw"
    
    return ''.join(['Case #',str(index+1),': ',str(result),'\n'])

def replaceTO(b):
    t=[]
    for r in b:
        tc = []
        t.append(tc)
        for c in r:
            if c == "T":
                tc.append("O")
            else:
                tc.append(c)
    return t

def replaceTX(b):
    t=[]
    for r in b:
        tc = []
        t.append(tc)
        for c in r:
            if c == "T":
                tc.append("X")
            else:
                tc.append(c)
    return t

def checkright(b,i,j,k):
    c = b[i][j]
    
    flag = True
    for os in range(1,k):
        if j+os >= len(b) or c != b[i][j+os]:
            flag = False
            break
    return flag

def checkdown(b,i,j,k):
    c = b[i][j]
    
    flag = True
    for os in range(1,k):
        if i+os >= len(b) or c != b[i+os][j]:
            flag = False
            break
    return flag

def checkdownright(b,i,j,k):
    c = b[i][j]
    
    flag = True
    for os in range(1,k):
        if i+os >= len(b) or j+os >= len(b) or c != b[i+os][j+os]:
            flag = False
            break
    return flag

def checkdownleft(b,i,j,k):
    c = b[i][j]
    
    flag = True
    for os in range(1,k):
        if i+os >= len(b) or j-os < 0 or c != b[i+os][j-os]:
            flag = False
            break
    return flag

if __name__ == "__main__":
    readFile('A-small-attempt0.in')
