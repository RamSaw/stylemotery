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
    for i in range(datasetSize):
        lineNr = lineNr + 1
        try:
            lineText = fileobject.readline()
        except:
            print("Failed to read line ", lineNr)
            sys.exit()
        NMarray = lineText.split(" ")
        N = int(NMarray[0])
        M = int(NMarray[1])
        lawnArray = []
        for j in range(N):
            lineNr = lineNr + 1
            try:
                lineText = fileobject.readline()
            except:
                print("Failed to read line ", lineNr)
                sys.exit()
            if lineText[-1] == "\n":
                textToParse = lineText[0:-1]
            else:
                textToParse = lineText
            lineArrayStrings = textToParse.split(" ")
            lineArrayInt = []
            for k in range(len(lineArrayStrings)):
                lineArrayInt.append(int(lineArrayStrings[k]))
            lawnArray.append(lineArrayInt)
        horizontalMaxArray = []
        for j in range(N):
            horizontalMaxArray.append(max(lawnArray[j]))
        verticalMaxArray = []
        for j in range(M):
            verticalMaxArray.append(0)
            for k in range(N):
                if verticalMaxArray[j] < lawnArray[k][j]:
                    verticalMaxArray[j] = lawnArray[k][j]
        possible = True
        for j in range(N):
            for k in range(M):
                if lawnArray[j][k] < horizontalMaxArray[j] and lawnArray[j][k] < verticalMaxArray[k]:
                    possible = False
                    break
            if not possible:
                break
        if possible:
            output = "YES"
        else:
            output = "NO"
        if i == 0:
            startCharacter = ""
        else:
            startCharacter = "\n"
        print(startCharacter, "Case #", i+1, ": ", output, end="", sep="")
