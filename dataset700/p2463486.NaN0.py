def isFair(aNumber):
    numberString = str(aNumber)
    numberCount = len(numberString)
    for i in range(math.ceil(numberCount/2)):
        if numberString[i] != numberString[numberCount - 1 - i]:
            return False
    return True

import math
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
    lineNr = 1
    fairArray = []
    for i in range(1, 10000001):
        if isFair(i):
            iSquare = i * i
            if isFair(iSquare):
                fairArray.append(i)
    for i in range(datasetSize):
        lineNr = lineNr + 1
        try:
            lineText = fileobject.readline()
        except:
            print("Failed to read line ", lineNr)
            sys.exit()
        ABarray = lineText.split(" ")
        A = int(ABarray[0])
        B = int(ABarray[1])
        aStart = math.ceil(math.sqrt(A))
        aEnd = math.floor(math.sqrt(B))
        fairNumbers = 0
        for j in range(len(fairArray)):
            fair = fairArray[j]
            if aStart > fair:
                continue
            if aEnd < fair:
                break
            fairNumbers = fairNumbers + 1
        if i == 0:
            startCharacter = ""
        else:
            startCharacter = "\n"
        print(startCharacter, "Case #", i+1, ": ", str(fairNumbers), end="", sep="")
