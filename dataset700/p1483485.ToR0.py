m = {' ': ' ', 'a': 'y', 'c': 'e', 'b': 'h', 'e': 'o', 'd': 's', 'g': 'v', 'f': 'c', 'i': 'd', 'h': 'x', 'k': 'i', 'j': 'u', 'm': 'l', 'l': 'g', 'o': 'k', 'n': 'b', 'q': 'z', 'p': 'r', 's': 'n', 'r': 't', 'u': 'j', 't': 'w', 'w': 'f', 'v': 'p', 'y': 'a', 'x': 'm', 'z': 'q'}

def readFile(file):
    name = file[:file.index('.')]
    ##print(name)
    f = open(file)
    fout = open(name+'.out','w')
    cases = int(f.readline().strip())
    for i in range(cases):
        text = f.readline().strip()
        result = execute(i,text)
        print(result)
        fout.write(result)

def execute(index,text):
    print((index,text))
    t = ''
    
    for c in text :
        t += m[c]
          
    return ''.join(['Case #',str(index+1),': ',str(t),'\n'])
        

if __name__ == "__main__":
    readFile('A-small-attempt0.in')
