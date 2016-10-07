mapping = {' ': ' ', 'a': 'y', 'c': 'e', 'b': 'h', 'e': 'o', 'd': 's',
           'g': 'v', 'f': 'c', 'i': 'd', 'h': 'x', 'k': 'i', 'j': 'u',
           'm': 'l', 'l': 'g', 'o': 'k', 'n': 'b', 'p': 'r', 's': 'n',
           'r': 't', 'u': 'j', 't': 'w', 'w': 'f', 'v': 'p', 'y': 'a',
           'x': 'm', 'z': 'q', 'q': 'z'}

def translate(s):
    return "".join([mapping[a] for a in s])

if __name__ == "__main__":
    T = int(input())
    for i in range(1, T+1):
        s = translate(input().strip())
        print("Case #%d: %s" %(i, s))

        
