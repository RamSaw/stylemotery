def solve(c, f, x):
    time = 0
    rate = 2

    while((x-c)/rate > x/(rate+f)):
        time += c/rate

        rate += f

    time += x/rate

    return time

def main(file):
    fin = open(file)
    fout = open(file[:-2]+"out", "w")

    t = int(fin.readline().strip())

    for i in range(t):
        (c,f,x) = tuple([float(x) for x in fin.readline().split()])

        result = solve(c,f,x)

        fout.write("Case #%d: %s\n" % (i+1, str(result)))

    fin.close()
    fout.close()

if __name__ == "__main__":
    #print(solve(500.0,4.0,2000.0))
    main("B-small-attempt1.in")
