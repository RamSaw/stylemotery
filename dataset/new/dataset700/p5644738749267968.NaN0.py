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
        blockCount = int(lineText[0:-1])
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
        blocksNaomi = sorted(list(map(float, textToParse.split(" "))))
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
        blocksKen = sorted(list(map(float, textToParse.split(" "))))
        pointsFairKen = 0
        pointsFairNaomi = 0
        j = 0
        for blockNaomi in blocksNaomi:
            while True:
                if blocksKen[j] > blockNaomi:
                    pointsFairKen += 1
                    j += 1
                    break
                else:
                    j += 1
                    if j == len(blocksKen):
                        break
            if j == len(blocksKen):
                break
        pointsFairNaomi = blockCount - pointsFairKen
        pointsUnfairKen = 0
        pointsUnfairNaomi = 0
        startPosKen = 0
        for j in range(blockCount):
            if blocksNaomi[j] < blocksKen[startPosKen]:
                pointsUnfairKen += 1 # remove Ken's heaviest block by high fake value
            else:
                # lightest block of Naomi heavier than lightest of Ken
                startPosKen += 1
                pointsUnfairNaomi += 1
        if i == 0:
            startCharacter = ""
        else:
            startCharacter = "\n"
        print(startCharacter, "Case #", i+1, ": ", pointsUnfairNaomi, " ", pointsFairNaomi, end="", sep="")