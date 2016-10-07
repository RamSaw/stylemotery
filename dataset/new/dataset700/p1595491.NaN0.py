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
        N = int(inputParams[0]) # number of Googlers
        S = int(inputParams[1]) # surprising result count
        p = int(inputParams[2]) # best value to reach
        tiArray = []
        for j in range(3, len(inputParams)):
            tiArray.append(int(inputParams[j]))
        tiArray.sort()
        tiArray.reverse()
        countP = 0
        for ti in tiArray:
            # every total voting number can consist of unsurprising votes
            #
            # the lowest total sum of the judging values with a best value of p
            # for an unsurprising judging result is p + (p-1) + (p-1) = 3p - 2
            if ti >= 3 * p - 2:
                countP = countP + 1
            # only total voting numbers of 2 and bigger can consist of
            # surprising votes
            #
            # the lowest total sum of the judging values with a best value of p
            # for a surprising judging result is p + (p-2) + (p-2) = 3p - 4
            #
            # because we started descending and with the unsurprising votes
            # we don't have to check for ti<=28
            elif ti >= 3 * p - 4 and ti >= 2:
                # can be still expected surprising votes?
                if S != 0:
                    countP = countP + 1
                    S = S - 1
                # if there are no more surprising votes possible, we can stop
                # here because the values get smaller and can't be created by
                # normal votes for p
                if S == 0:
                    # because of the descending sorting order, only surprising
                    # judging results are still possible, but we already used
                    # the maximum amount for these
                    break
        if i == 0:
            startCharacter = ""
        else:
            startCharacter = "\n"
        print(startCharacter, "Case #", i+1, ": ", countP, end="", sep="")