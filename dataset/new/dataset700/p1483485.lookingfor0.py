inps = ["ejp mysljylc kd kxveddknmc re jsicpdrysi", "rbcpc ypc rtcsra dkh wyfrepkym veddknkmkrkcd", "de kr kd eoya kw aej tysr re ujdr lkgc jv"]
outs = ["our language is impossible to understand","there are twenty six factorial possibilities", "so it is okay if you want to just give up"]

d = {'z':'q', 'q':'z'}

for i in range(3):
    inp, out = inps[i], outs[i]
    for j in range(len(inp)):
        d[inp[j]] = out[j]

n = int(input())
for i in range(n):
    s = input()
    print("Case #%d:" % (i+1), "".join([d[c] for c in s]))