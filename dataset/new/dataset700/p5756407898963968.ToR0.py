def solve(ans1, lay1, ans2, lay2):

    pos1 = lay1[ans1-1]
    pos2 = lay2[ans2-1]

    pos_set = [c for c in pos1 if c in pos2]

    result = ""
    if len(pos_set)==1:
        result = str(pos_set.pop())
    elif len(pos_set)==0:
        result = "Volunteer cheated!"
    elif len(pos_set) > 1:
        result = "Bad magician!"

    return result

def main(file):

    f = open(file)
    fout = open(file[:-2]+"out","w")

    n = int(f.readline().strip())

    for i in range(n):
        ans1 = int(f.readline().strip())
        lay1 = []
        for j in range(4):
            row = [int(x) for x in f.readline().split()]
            lay1.append(row)

        ans2 = int(f.readline().strip())
        lay2 = []
        for k in range(4):
            row = [int(x) for x in f.readline().split()]
            lay2.append(row)

        result = solve(ans1, lay1, ans2, lay2)
        fout.write("Case #%d: %s\n" % (i+1, result))

    f.close()
    fout.close()

if __name__ == "__main__":
    main("A-small-attempt0.in")
