def solve(r, c, m):

    size = r*c
    free = size - m
    if size == m:
        #impossible
        return None 

    elif free == 1:
        result = []
        for i in range(r):
            result.append(["*"]*c)
        result[0][0] = "c"
        return result

    elif m == 0:
        result = []
        for i in range(r):
            result.append(["."]*c)
        result[0][0] = "c"
        return result

    elif r == 1 or c == 1:
        result = []
        for i in range(r):
            result.append(["."]*c)

        if r==1:
            for j in range(m):
                result[0][j] = "*"
            result[0][-1] = "c"
        elif c==1:
            for k in range(m):
                result[k][0] = "*"
            result[-1][0] = "c"

        return result     

    else:
        result = []
        for i in range(r):
            result.append(["*"]*c)

        import math

        dimX = round(math.sqrt(free))
        dimY = math.ceil(free/dimX)
        take = (dimX * dimY) - free

        #print(free, dimX, dimY, take)
        
        if dimY > c or dimX > r:
            #change dim

            if dimY <= r and dimX <= c:
                #transpose
                (dimX,dimY) = (dimY,dimX)

            else:
                minDim = min(c, r)
                dimX = minDim
                dimY = math.ceil(free/dimX)
                take = (dimX * dimY) - free

                if (dimY > c or dimX > r) and (dimY <= r and dimX <= c):
                    #transpose 
                    (dimX,dimY) = (dimY,dimX)

        for i in range(dimX):
            for j in range(dimY):
                result[-i-1][-j-1] = "."

        #take
        diag = 0
        tx = 0
        ty = 0
        for k in range(take):
            result[-dimX+tx][-dimY+ty] = "*"

            if dimY < dimX:
                #add new row first
                if tx==0:
                    tx=ty+1
                    ty=0
                else:
                    ty += 1
                    tx -= 1
            else:
                if ty==0:
                    ty=tx+1
                    tx=0
                else:
                    ty -= 1
                    tx += 1

        #check row
        for row in range(r):
            count_free = 0
            for col in range(c):
                if result[row][col] == ".":
                    count_free += 1

            if count_free == 1: #fault
                print((printResult(result)))
                print((r,c,m,"row check"))
                return None

        #check col
        for col in range(c):
            count_free = 0
            for row in range(r):
                if result[row][col] == ".":
                    count_free += 1
            
            if count_free == 1: #fault
                print((printResult(result)))
                print((r,c,m,"col check"))
                return None
            
        result[-1][-1] = "c"
        
        return result


def printResult(result):
    if not result:
        return "Impossible"
    else:
        t = []
        for l in result:
            t.append("".join(l))
        return "\n".join(t)

def main(file):
    f = open(file)
    fout = open(file[:-2]+"out", "w")

    t = int(f.readline().strip())

    for i in range(t):
        (r, c, m) = tuple([int(x) for x in f.readline().split()])
        result = printResult(solve(r, c, m))

        fout.write("Case #%d:\n" % (i+1,))
        fout.write(result+"\n")

    f.close()
    fout.close()


if __name__ == "__main__":
    main("C-small-attempt2.in")

    #print(printResult(solve(2,5,5)))
