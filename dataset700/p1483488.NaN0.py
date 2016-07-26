import sys

if len(sys.argv) == 1:
    print("No input file provided.")
    sys.exit()
else:
    filename = sys.argv[1]
    try:
        fileobject = open(filename, 'r')
    except:
        print("Failed to open given file.")
        sys.exit()
    try:
        firstLine = fileobject.readline()
    except:
        print("Failed to read first line.")
        sys.exit()
    datasetSize = int(firstLine)
    if not datasetSize:
        print("Unable to parse dataset size.")
        sys.exit()
    for i in range(datasetSize):
        try:
            lineText = fileobject.readline()
        except:
            print("Failed to read line ", i + 2)
            sys.exit()
        if lineText[-1] == "\n":
            textToParse = lineText[0:-1]
        else:
            textToParse = lineText
        inputParams = textToParse.split(" ")
        A = int(inputParams[0])
        B = int(inputParams[1])
        digitCount = len(str(A))
        pairCount = 0
        if digitCount != 1:
            # go only up to B-1 because B is the maximum, but with n=B
            # m would have to be bigger than B
            for n in range(A, B):
                pairArray = []
                strN = str(n)
                for j in range(1, digitCount):
                    if strN[j] != "0":
                        m = int(strN[j:] + strN[:j])
                        if A <= m and m <=B and m>n and m not in pairArray:
                            pairCount = pairCount + 1
                            pairArray.append(m)
        if i == 0:
            startCharacter = ""
        else:
            startCharacter = "\n"
        print(startCharacter, "Case #", i+1, ": ", pairCount, end="", sep="")