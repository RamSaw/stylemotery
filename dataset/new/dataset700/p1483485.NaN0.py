import sys

googlereseArray = ["y qee",
"ejp mysljylc kd kxveddknmc re jsicpdrysi",
"rbcpc ypc rtcsra dkh wyfrepkym veddknkmkrkcd",
"de kr kd eoya kw aej tysr re ujdr lkgc jv"]
plaintextArray = ["a zoo",
"our language is impossible to understand",
"there are twenty six factorial possibilities",
"so it is okay if you want to just give up"]
abc = "abcdefghijklmnopqrstuvwxyz"
plaintextLettersArray = list(abc)
googlereseLettersArray = list(abc)
decipherDict = {}
i = 0
for googlereseSentence in googlereseArray:
    j = 0
    for letter in googlereseSentence:
        if letter != " " and letter not in decipherDict:
            decipherDict[letter] = plaintextArray[i][j]
            googlereseLettersArray.remove(letter)
            plaintextLettersArray.remove(plaintextArray[i][j])
        j = j + 1
    i = i + 1
# q has not been been translated to googlerese, but all other letters
# so it's the only letter remaining (and its translation in the other array)
decipherDict[googlereseLettersArray[0]] = plaintextLettersArray[0]
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
        lineTextPlain = ""
        for letter in textToParse:
            if letter == " ":
                lineTextPlain = lineTextPlain + " "
            else:
                lineTextPlain = lineTextPlain + decipherDict[letter]
        if i == 0:
            startCharacter = ""
        else:
            startCharacter = "\n"
        print(startCharacter, "Case #", i+1, ": ", lineTextPlain, end="", sep="")