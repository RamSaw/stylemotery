# -*- coding: utf-8 -*-

t = int(input())

d = {'a': 'y', 'c': 'e', 'b': 'h', 'e': 'o', 'd': 's', 'g': 'v', 'f': 'c', 'i': 'd', 'h': 'x', 'k': 'i', 'j': 'u', 'm': 'l', 'l': 'g', 'o': 'k', 'n': 'b', 'p': 'r', 's': 'n', 'r': 't', 'u': 'j', 't': 'w', 'w': 'f', 'v': 'p', 'y': 'a', 'x': 'm', 'q':'z', 'z':'q'}

for i in range(1, t+1):
    res = []
    line = input().split(' ')
    for e in line:
        res.append(''.join([d[a] for a in e]))
    print('Case #%d: ' % i+ ' '.join(res)) 
