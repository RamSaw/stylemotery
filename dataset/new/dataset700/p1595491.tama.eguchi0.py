#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Problem B. Dancing With the Googlers
# http://code.google.com/codejam/contest/1460488/dashboard#s=p1
#

import sys
import string



def solve(S, p, tlist):

	above = 0
	consider = 0

	for t in tlist:
		avg = t / 3
		mod = t % 3

		if mod == 0:
			
			if avg >= p:
				above += 1
			elif avg + 1 >= p and t > 0:
			
				consider += 1

		elif mod == 1:
			
			if avg + 1 >= p:
				above += 1
			
			

		elif mod == 2:
			
			if avg + 1 >= p:
				above += 1
			elif avg + 2 >= p:
			
				consider += 1

	return above + min(S, consider)


def main(IN, OUT):
	N = int(IN.readline())
	for index in range(N):
		data = list(map(int, IN.readline().strip().split()))
		(N, S, p), tlist = data[:3], data[3:]
		OUT.write('Case #%d: %d\n' % (index + 1, solve(S, p, tlist)))


if __name__ == '__main__':
	main(sys.stdin, sys.stdout)

