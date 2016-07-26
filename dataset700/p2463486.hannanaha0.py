from gcjbase import *
from decimal import *

def read_input(filename):
    data = []
    with open(filename, "r") as f:
        cases = read_ints(f)[0]
        # =============================================
        for _ in range(cases):
            line = read_ints(f)
            data.append({"A": line[0], "B": line[1]})
        # =============================================
    return data

def make_output(fname, output):
    CASE_PRFX = "Case #%s: "
    fname = fname + time.strftime("%H%M%S") + ".out"
    with open(fname, "w") as f:
        # =============================================
        restext = []
        print("Output content ===============")
        for i, v in enumerate(output):
            line = CASE_PRFX % (i+1,) + str(v) + "\n"
            print(line[:-1])
            restext.append(line)
        print("=" * 30)
        f.writelines(restext)
        # =============================================

# ----------------------------------------------------------------------

@memoizeit
def nextpaly(numstr):
    """ find the next palyndrom for a given one """
    if not numstr:
        return numstr
    if all([c == '9' for c in numstr]):
        return str(1) + str(0) * (len(numstr) - 1) + str(1)
    isodd = len(numstr) % 2
    pivot = len(numstr) / 2
    if isodd:
        npivot = int(numstr[pivot])
        if npivot < 9:
            return numstr[:pivot] + str(npivot + 1) + numstr[pivot+1:]
        shell = nextpaly(numstr[:pivot] + numstr[pivot+1:])
        return shell[:pivot] + str(0) + shell[pivot:]
    shell = nextpaly(numstr[:pivot] + numstr[pivot+1:])
    return shell[:pivot] + str(shell[pivot-1]) + shell[pivot:]


def ispaly(num):
    snum = str(num)
    ll = len(snum)
    for i in range(ll/2):
        if snum[i] != snum[ll-i-1]:
            return False
    return True

def closestpaly(num):
    while not ispaly(num):
        num += 1
    return num

@timeit
def solveit(case):
    print(case)
    A = Decimal(case["A"])
    B = Decimal(case["B"])
    
    baseroot = A.sqrt().to_integral_value(ROUND_CEILING)
    baseroot = Decimal(closestpaly(baseroot))

    count = 0
    p = baseroot
    while True:
        sq = p * p
        if sq > B:
            return count
        if ispaly(sq):
            count += 1
        p = Decimal(nextpaly(str(p)))
    
    return count

@timeit
def main(fname):
    data = read_input(fname)
    output = []
    for i, case in enumerate(data):
        # =============================================
        with localcontext() as ctx:
            ctx.prec = 101
            res = solveit(case)
            output.append(res)
        # =============================================
    make_output(fname, output)


if __name__ == '__main__':
#    main("sample.in")
#    main("test.in")
    main("small.in")
#    main("large.in")