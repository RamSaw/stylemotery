from sys import stdin
import math

pots = [int(10**k) for k in range(0,16)]

T = int(stdin.readline())

for i in range(1,T+1):
	a,b = list(map(int, stdin.readline().split()))
	t = 0
	m = len(str(a))

	pp = int(10**m)

	for n in range(a,b):
		ss = []
		q = 1
		p = pp
		for j in range(1,m):
			q *= 10
			p /= 10
			r = (n % p) * q + (n /p)
			if n < r and r <= b and not r in ss: 
				ss.append(r)
				t += 1


	print("Case #%d: %d" % (i,t))

