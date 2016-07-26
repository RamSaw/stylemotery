def readFile(file):
    name = file[:file.index('.')]
    ##print(name)
    f = open(file)
    fout = open(name+'.out','w')
    cases = int(f.readline().strip())
    for i in range(cases):
        case = f.readline().strip().split()
        n = int(case[0])
        s = int(case[1])
        p = int(case[2])
        scores = [int(x) for x in case[3:]]
        result = execute(i,n,s,p,scores)
        print(result)
        fout.write(result)

def execute(index,n,s,p,scores):
    print((index,n,s,p,scores))
    count=0
    tempS = s
    triplets = []

    for j in range(n):
        mean = int(scores[j]/3)
        remainder = scores[j]%3

        triplet = [mean for x in range(3)]

        c = 0
        while remainder > 0 :
            triplet[c] = triplet[c]+1
            remainder-=1
            c+=1

        triplets.append(triplet)

    ##add suprising
    for t in triplets:
        if t[0] >= p:
            count+=1
            continue
        elif (tempS > 0 and (p-t[0]) == 1 and t[0] == t[1] and t[1] > 0):
            #surprise case
            t[0]=t[0]+1
            t[1]=t[1]-1
            count+=1
            tempS-=1
            continue

    print(tempS)
    print(triplets)
    
    return ''.join(['Case #',str(index+1),': ',str(count),'\n'])
        


if __name__ == "__main__":
    readFile('B-small-attempt1.in')
