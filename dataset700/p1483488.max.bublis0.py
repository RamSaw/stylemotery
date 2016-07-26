"""
Google Code Jam 2012 Problem C
Usage:
    python problem_c.py < input.txt > output.txt
"""
import itertools
import sys


def is_recycled(n, m, _cache={}, _cache_sorted={}, _cache_length={}):
    if not n in _cache:
        _cache[n] = str(n)

    if not m in _cache:
        _cache[m] = str(m)

    str_n = _cache[n]
    str_m = _cache[m]

    if not n in _cache_sorted:
        _cache_sorted[n] = sorted(str_n)

    if not m in _cache_sorted:
        _cache_sorted[m] = sorted(str_m)

    if _cache_sorted[n] != _cache_sorted[m]:
        return False

    if not n in _cache_length:
        _cache_length[n] = len(str_n)

    for i in range(_cache_length[n] + 1):
        if str_m == str_n[i:] + str_n[:i]:
            return True

    return False


def solve_problem():
    number_of_cases = int(sys.stdin.readline())

    for i in range(1, number_of_cases + 1):
        case = sys.stdin.readline().strip()
        A, B = list(map(int, case.split()))
        result = sum(map(lambda n_m: is_recycled(n_m[0], n_m[1]),
                                    itertools.combinations(range(A, B + 1), 2)))

        sys.stdout.write('Case #{0}: {1}\n'.format(i, result))

if __name__ == '__main__':
    solve_problem()
