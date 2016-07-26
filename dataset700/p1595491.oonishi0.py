# coding: shift-jis

import sys
f = file("B-small-attempt0.in")
#f = file("test.in")
#w = sys.stdout
w = file("answer.txt", "w")
cnt = int(f.readline()[:-1])
for no in range(cnt):
	l = f.readline()[:-1].split()
	T, s, p = list(map(int, l[:3]))
	ts = list(map(int, l[3:]))
	ns = p*3-2 if p*3-2 > 0 else 0
	ss = p*3-4 if p*3-4 > 0 else 31
	l = [x for x in ts if x<ns]
	c = min([len([x for x in l if x>=ss]), s])
	
	print("Case #%d:"%(no+1), T-len(l)+c, file=w)


