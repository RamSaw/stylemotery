#! /usr/bin/python3
ncases = int(eval(input()))

for n in range(1, ncases+1):
    row = int(eval(input()))
    for m in range(1, 5):
        if m == row:
            r1list = [int(x) for x in input().split()]
            assert len(r1list) == 4
        else:
            tmp = eval(input())
    row = int(eval(input()))
    for m in range(1,5):
        if m == row:
            r2list = [int(x) for x in input().split()]
            assert len(r1list) == 4
        else:
            tmp = eval(input())
    nset = set(r1list) & set(r2list)
    if len(nset) == 1:
        print(("Case #%d:" % n, nset.pop()))
    elif len(nset) > 1:
        print(("Case #%d:" % n, "Bad magician!"))
    else:
        print(("Case #%d:" % n, "Volunteer cheated!"))
