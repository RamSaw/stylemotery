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
        if lineText[-1] == "\n":
            textToParse = lineText[0:-1]
        else:
            textToParse = lineText
        inputParams = textToParse.split(" ")
        R = int(inputParams[0]) # rows
        C = int(inputParams[1]) # columns
        M = int(inputParams[2]) # mines
        if i == 0:
            startCharacter = ""
        else:
            startCharacter = "\n"
        print(startCharacter, "Case #", i + 1, ":", end="", sep="")
        if R == 1 or C == 1:
            # This is one dimensional and always solvable. Fill up the line from one end.
            if R == 1:
                print("\n", "*" * M, "." * (C - M - 1), "c", end="", sep="")
            else:
                for j in range(M):
                    print("\n*", end="")
                for j in range(R - M - 1):
                    print("\n.", end="")
                print("\nc", end="")
        else:
            # Two dimensions:
            # Without further restrictions: All non-mine fields will be revealed if they are
            # horizontal, vertical or diagonal adjacent to a field which has no mine as neighbor.
            # We reduce the available area and number of mines by filling whole lines on the edge
            # of the available area with mines while also trying to reduce it to a rectangle of
            # width and length 4.
            Rred = R
            Cred = C
            Mred = M
            while (Rred > 3 or Cred > 3) and Mred > 0:
                if Rred > Cred:
                    if Mred >= Cred:
                        Mred -= Cred
                        Rred -= 1
                    else:
                        break
                else:
                    if Mred >= Rred:
                        Mred -= Rred
                        Cred -= 1
                    else:
                        break
            # For areas with side lengths of more than 4 which can't be reduced further, the
            # problem is always solvable the number of mines available is shorter than the length
            # of the longer side and the mines can be placed on two adjacent lines.
            # E.g. a 5x5 square with 9 or more mines could be reduced to a 4x4 square like this:
            # ~ = mine in area removed by reduction or edge, ? = reduced area
            # ~~~~~~ => ~~~~~~   For <9 mines in a 5x5: ~~~~~~
            # ~?????    ~#####                          ~#####
            # ~?????    ~#????                          ~###..
            # ~?????    ~#????                          ~.....
            # ~?????    ~#????                          ~.....
            # ~?????    ~#????                          ~.....
            if min(Cred, Rred) >= 3 and max(Cred, Rred) > 3:
                # Problem can be solved
                for j in range(R - Rred):
                    print("\n", "*" * C, end="", sep="")
                # Find longer side:
                if Rred > Cred:
                    for j in range(Rred):
                        print("\n", "*" * (C - Cred), end="", sep="")
                        if Mred > 0:
                            print("*", end="")
                            Mred -= 1
                            if Mred > 0 and j != Rred - 2 and Cred > 3:
                                # Don't create two niches in the last line
                                print("*", end="")
                                print("." * (Cred - 2), end="")
                                Mred -= 1
                            else:
                                if j == Rred - 1:
                                    # last line, insert click in bottom right
                                    print("." * (Cred - 2), "c", end="", sep="")
                                else:
                                    print("." * (Cred - 1), end="")
                        else:
                            if j == Rred - 1:
                                # last line, insert click in bottom right
                                print("." * (Cred - 1), "c", end="", sep="")
                            else:
                                print("." * Cred, end="")
                else:
                    for j in range(Rred):
                        print("\n", "*" * (C - Cred), end="", sep="")
                        if j == 0:
                            if Mred >= Cred:
                                print("*" * Cred, end="")
                                Mred -= Cred
                            else:
                                minesInLine = min(Mred, Cred - 2) # Don't create a niche
                                print("*" * minesInLine, end="")
                                Mred -= minesInLine # Don't create a niche
                                print("." * (Cred - minesInLine), end="")
                        elif j == 1: # second line
                            print("*" * Mred, "." * (Cred - Mred), end="", sep="")
                        elif j != Rred - 1:
                            print("." * Cred, end="")
                        else:
                            print("." * (Cred - 1), "c", end="", sep="")
                        
            elif Rred == 3 and Cred == 3:
                # Solutions by logical thinking:
                # ~ = mine in area removed by reduction or edge
                #
                # Mred  Mred  Mred  Mred Mred
                # 8     5     3     1    0
                # ~~~~  ~~~~  ~~~~  ~~~~ ~~~~
                # ~###  ~###  ~###  ~#.. ~...
                # ~###  ~#..  ~...  ~... ~...
                # ~##c  ~#.c  ~..c  ~..c ~..c
                # For 7, 6, 4 or 2 remaining mines in a 3x3 square, there is no solution for the issue
                # because either there is only position without a mine (Mred == 1) or for opening all
                # at once, there have to be positions without adjacent mines. These require mine-free
                # areas of 4 (corner), 6 (edge) or 9 positions (no contact to edge). Extending the area
                # works only in steps of 2 (corner or edge along edge), 3 horizontal or vertical but
                # not along the edge, or 5 (diagonal). With this restricted area, other unmined area
                # sizes can't be built
                if Mred not in [8, 5, 3, 1, 0]:
                    print("\nImpossible", end="")
                else:
                    for j in range(R - Rred):
                        print("\n", "*" * C, end="", sep="")
                    for j in range(Rred):
                        print("\n", "*" * (C - Cred), end="", sep="")
                        minesInLine = 0
                        if Mred >= Cred:
                            print("*" * Cred, end="")
                            minesInLine = Cred
                        else:
                            if j != Rred - 1:
                                minesInLine = min(Mred, Cred - 2) # Don't create a niche
                                print("*" * minesInLine, end="")
                                print("." * (Cred - minesInLine), end="")
                            else:
                                print("*" * Mred, "." * (Cred - Mred - 1), "c", end="", sep="")
                        Mred -= minesInLine
            elif Rred == 2 or Cred == 2:
                if Mred % 2 == 1: # odd numbers force a niche
                    print("\nImpossible", end="")
                else:
                    # Find longer side:
                    if Rred > Cred:
                        for j in range(R - Rred):
                            print("\n**", end="", sep="")
                        for j in range(Rred):
                            if j == Rred - 1:
                                # last line, insert click in bottom right
                                print("\n.c", end="")
                            else:
                                print("\n..", end="")
                    else:
                        minesInLine = Mred // 2
                        for j in range(Rred):
                            print("\n", "*" * (C - Cred), end="", sep="")
                            if j == 0:
                                print("*" * minesInLine, "." * (Cred - minesInLine), end="", sep="")
                            elif j == 1: # second line
                                print("*" * minesInLine, "." * (Cred - minesInLine - 1), "c", end="", sep="")
                            
