from codejam import *

for i in range(readint()):
    srow = readint()    
    rows = [None] * 4
    for j in range(4):
        rows[j] = set(readintarray())
    
    srow2 = readint()    
    rows2 = [None] * 4
    for j in range(4):
        rows2[j] = set(readintarray())
    
    matches = rows[srow - 1].intersection(rows2[srow2 - 1])
    if len(matches) == 0:
        print("Case #%d: Volunteer cheated!" % (i + 1))
    elif len(matches) == 1:
        print("Case #%d: %d" % (i + 1, matches.pop()))
    else:
        print("Case #%d: Bad magician!" % (i + 1))
