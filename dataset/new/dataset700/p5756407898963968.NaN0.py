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
        lineWithNrSetA = int(lineText[0:-1])
        lineNr += 1
        setA = []
        for j in range(4):
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
            setA.append(set(textToParse.split(" ")))
        lineText = fileobject.readline()
        lineWithNrSetB = int(lineText[0:-1])
        lineNr += 1
        setB = []
        for j in range(4):
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
            setB.append(set(textToParse.split(" ")))
        sharedElements = setA[lineWithNrSetA - 1] & setB[lineWithNrSetB - 1]
        if len(sharedElements) == 0:
            output = "Volunteer cheated!"
        elif len(sharedElements) >= 2:
            output = "Bad magician!"
        else: # len(sharedElements) == 1
            output = sharedElements.pop()
        if i == 0:
            startCharacter = ""
        else:
            startCharacter = "\n"
        print(startCharacter, "Case #", i+1, ": ", output, end="", sep="")
