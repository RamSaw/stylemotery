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
        textToParse = lineText[0:-1]
        inputParams = textToParse.split(" ")
        C = float(inputParams[0]) # Cost of cookie farm in cookies
        F = float(inputParams[1]) # Cookie growth gain per farm in cookies/second
        X = float(inputParams[2]) # Cookies required to win game
        Cgrowth = 2
        Ccurrent = 0
        Tpassed = 0
        while True:
            TfinishWithoutBuy = (X - Ccurrent) / Cgrowth
            TuntilBuy = (C - Ccurrent) / Cgrowth
            TfinishAfterBuy = X / (Cgrowth + F) # Ccurrent = 0 after farm purchase
            if TuntilBuy + TfinishAfterBuy < TfinishWithoutBuy:
                Cgrowth += F
                Ccurrent = 0
                Tpassed += TuntilBuy
            else:
                Tpassed += TfinishWithoutBuy
                break
        if i == 0:
            startCharacter = ""
        else:
            startCharacter = "\n"
        print(startCharacter, "Case #", i+1, ": %.7f" % round(Tpassed, 7), end="", sep="")