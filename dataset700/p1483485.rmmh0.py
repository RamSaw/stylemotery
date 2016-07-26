mapping = {'y':'a', 'e':'o', 'q':'z', 'z':'q'}

for src, dst in [("ejp mysljylc kd kxveddknmc re jsicpdrysi", "our language is impossible to understand"),
    ("rbcpc ypc rtcsra dkh wyfrepkym veddknkmkrkcd","there are twenty six factorial possibilities"),
    ("de kr kd eoya kw aej tysr re ujdr lkgc jv","so it is okay if you want to just give up")]:
    for a, b in zip(src,dst):
        mapping[a] = b

n_cases = eval(input())


for case in range(1, n_cases + 1):
    string = input()

    out = ''.join(mapping.get(c, c) for c in string)
            
    print("Case #%d: %s" % (case, out))
