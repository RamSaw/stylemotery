fi=open("A-small-attempt0.in",'r')#Input File
#fi=open("A.in",'r')#Input File
fo=open("A-small-attempt0.out","w")#Output File
#fo=open("A.out","w")#Output File

mapp = dict(a='y', b='h', c='e', d='s', e='o', f='c', g='v', h='x', i='d', j='u', k='i', l='g', m='l', n='b', o='k', p='r', q='z', r='t', s='n', t='w', u='j', v='p', w='f', x='m', y='a', z='q')
T=int(fi.readline())
for case in range(1,T+1,1):
    ans = ""
    for ch in fi.readline():
        ans += mapp[ch] if ch in mapp else ch
    #print ans
    fo.write("Case #%s: %s"%(case, ans))
