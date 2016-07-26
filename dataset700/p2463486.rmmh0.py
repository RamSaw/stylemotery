import sys
#sys.stdin = open("c_example.in")

n_cases = eval(input())

def to_ints(s):
    return list(map(int, s.split()))

def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

for case in range(1, n_cases + 1):
    a, b = to_ints(input())

    nums = list(range(int(b ** .5) + 2))
    palins = list(filter(is_palindrome, nums))
    squares = [x**2 for x in palins]
    palin_squares = list(filter(is_palindrome, squares))
    range_squares = [x for x in palin_squares if a <= x <= b]

    print("Case #%d: %s" % (case, len(range_squares)))
