import decimal

from base import *

def read_input(filename):
    data = []
    with open(filename, "r") as f:
        cases = read_ints(f)[0]
        # =============================================
        for _ in range(cases):
            case = {}
            case["first_guess"], = read_ints(f)
            case["first_setup"] = Mtrx.read_int_matrix(f, 4, 4)
            case["second_guess"], = read_ints(f)
            case["second_setup"] = Mtrx.read_int_matrix(f, 4, 4)
            data.append(case)
        # =============================================
    return data

def make_output(fname, output):
    CASE_PRFX = "Case #%s: "
    fname = fname + time.strftime("%H%M%S") + ".out"
    with open(fname, "w") as f:
        restext = []
        print("Output content ===============")
        # =============================================
        for i, outdata in enumerate(output):
            line = CASE_PRFX % (i + 1,) + str(outdata) + EOL
            print(line, end=' ')
            restext.append(line)
        # =============================================
        print("=" * 30)
        f.writelines(restext)

# ----------------------------------------------------------------------

CHEAT = "Volunteer cheated!"
BAD = "Bad magician!"

@timeit
def solveit(case):
    candidates = case["first_setup"].getrow(case["first_guess"] - 1)
    suspects = case["second_setup"].getrow(case["second_guess"] - 1)
    matches = [c for c in candidates if c in suspects]
    res = ""
    if len(matches) == 0:
        res = CHEAT
    elif len(matches) == 1:
        res = str(matches[0])
    else:
        res = BAD
    return res, candidates, suspects, matches

@timeit
def main(fname):
    data = read_input(fname)
    output = []
    for case in data:
        statreset() # reset cache stats
        # =============================================
        res = solveit(case)[0]
        output.append(res)
        # =============================================
    make_output(fname, output)


if __name__ == '__main__':
#    main("sample.in")
    main("A-small-attempt1.in")
#    main("B-large.in")
#    main("B-small-attempt0.in")
#    main("A-large.in")