fin = open('B-small-attempt0.in', 'r')
fout = open('B-output.txt', 'w')

cases = int(fin.readline()[:-1])

for case in range(cases) :
    line = list(map(int, fin.readline()[:-1].split(' ')))
    N, S, p = line[:3]
    T = line[3:]
    okLimit = p + 2*max(p-1,0)
    okIfSLimit = p + 2*max(p-2,0)
    ok = len([x for x in T if x >= okLimit])
    okIfS = len([x for x in T if okLimit > x >= okIfSLimit])
    res = ok + min(okIfS, S)
    # print 'Case #' + str(case+1) + ': ' + str(res)
    fout.write('Case #' + str(case+1) + ': ' + str(res) + '\n')
    
fin.close()
fout.close()
