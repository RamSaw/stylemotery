import string
n = int(input())
S = """
y qee
ejp mysljylc kd kxveddknmc re jsicpdrysi
rbcpc ypc rtcsra dkh wyfrepkym veddknkmkrkcd
de kr kd eoya kw aej tysr re ujdr lkgc jv
z
"""
T = """
a zoo
our language is impossible to understand
there are twenty six factorial possibilities
so it is okay if you want to just give up
q
"""
trans = {}
for i in range(len(S)):
    trans[S[i]] = T[i]
A = ""
B = ""
for (c, d) in list(trans.items()):
    A += c
    B += d
for i in range(n):
    print("Case #%d: %s" % (i + 1, input().strip().translate(string.maketrans(A, B))))
