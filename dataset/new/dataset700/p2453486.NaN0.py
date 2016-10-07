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
        boardArray = []
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
            boardArray.append(textToParse)
        if i < datasetSize - 1:
            # ignore the empty line between two data sets
            lineNr = lineNr + 1
            fileobject.readline()
        # horizontal check
        gameDone = True
        winner = None
        for j in range(4):
            emptyPos = False
            Oonly = True
            Xonly = True
            for k in range(4):
                if boardArray[j][k] == "O":
                    Xonly = False
                if boardArray[j][k] == "X":
                    Oonly = False
                if boardArray[j][k] == ".":
                    emptyPos = True
                    Oonly = False
                    Xonly = False
            if emptyPos:
                gameDone = False
            if Oonly:
                winner = "O"
            if Xonly:
                winner = "X"
        # vertical check
        if not winner:
            for j in range(4):
                emptyPos = False
                Oonly = True
                Xonly = True
                for k in range(4):
                    if boardArray[k][j] == "O":
                        Xonly = False
                    if boardArray[k][j] == "X":
                        Oonly = False
                    if boardArray[k][j] == ".":
                        emptyPos = True
                        Oonly = False
                        Xonly = False
                if emptyPos:
                    gameDone = False
                if Oonly:
                    winner = "O"
                if Xonly:
                    winner = "X"
        # diagonal check
        if not winner:
            # from (0,0) to (3,3)
            emptyPos = False
            Oonly = True
            Xonly = True
            for j in range(4):
                if boardArray[j][j] == "O":
                    Xonly = False
                if boardArray[j][j] == "X":
                    Oonly = False
                if boardArray[j][j] == ".":
                    emptyPos = True
                    Oonly = False
                    Xonly = False
            if emptyPos:
                gameDone = False
            if Oonly:
                winner = "O"
            if Xonly:
                winner = "X"
        # from (0,3) to (3,0)
        if not winner:
            emptyPos = False
            Oonly = True
            Xonly = True
            for j in range(4):
                if boardArray[j][3 - j] == "O":
                    Xonly = False
                if boardArray[j][3 - j] == "X":
                    Oonly = False
                if boardArray[j][3 - j] == ".":
                    emptyPos = True
                    Oonly = False
                    Xonly = False
            if emptyPos:
                gameDone = False
            if Oonly:
                winner = "O"
            if Xonly:
                winner = "X"
        if not winner:
            if not gameDone:
                output = "Game has not completed"
            else:
                output = "Draw"
        else:
            output = winner + " won"
        if i == 0:
            startCharacter = ""
        else:
            startCharacter = "\n"
        print(startCharacter, "Case #", i+1, ": ", output, end="", sep="")
