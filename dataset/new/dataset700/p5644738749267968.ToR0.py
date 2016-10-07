def solve_war(nlist, klist):

    sorted_nlist = sorted(nlist)
    sorted_klist = sorted(klist)

    score = 0

    while sorted_nlist:
        n = sorted_nlist.pop(0)
        k = 0
        for x in sorted_klist:
            if x > n:
                k = x
                break

        #print(n,k)

        if k != 0:
            sorted_klist.remove(k)
        else:
            score += 1
            sorted_klist.pop(0)

    return score

def solve_deceit(nlist, klist):

    sorted_nlist = sorted(nlist)
    sorted_klist = sorted(klist)

    score = 0

    while sorted_nlist:
        n = sorted_nlist.pop(0)
        min_k = sorted_klist[0]
        max_k = sorted_klist[-1]

        if n > min_k:
            score += 1
            sorted_klist.remove(min_k)
        else:
            sorted_klist.remove(max_k)
    
        #print(n,min_k, max_k)

    return score

def main(file):
    f = open(file)
    fout = open(file[:-2]+"out", "w")

    t = int(f.readline().strip())
    for i in range(t):
        n = int(f.readline().strip())

        nlist = [float(x) for x in f.readline().split()]
        klist = [float(x) for x in f.readline().split()]

        score_deceit = solve_deceit(nlist, klist)
        score_war = solve_war(nlist, klist)

        fout.write("Case #%d: %d %d\n" % (i+1, score_deceit, score_war))

    f.close()
    fout.close()

if __name__ == "__main__":
    #nlist = [0.186, 0.389, 0.907, 0.832, 0.959, 0.557, 0.300, 0.992, 0.899]
    #klist = [0.916, 0.728, 0.271, 0.520, 0.700, 0.521, 0.215, 0.341, 0.458]
    #print(solve_deceit(nlist,klist))
    main("D-small-attempt0.in")
    
