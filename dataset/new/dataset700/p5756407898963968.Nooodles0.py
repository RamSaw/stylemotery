T = int(input());
for q in range(T):
    R1 = int(input());
    Data1 = [];
    for i in range(4):
        Data1.append( list(map(int, input().split())) );
        
    R2 = int(input());
    Data2 = [];
    for i in range(4):
        Data2.append( list(map(int, input().split())) );

    Ans = [];
    for entry in Data1[R1-1]:
        if entry in Data2[R2-1]:
            Ans.append(entry);

    print("Case #%d:" % (q+1), end=' ');

    if len(Ans) == 0:
        print("Volunteer cheated!");
    if len(Ans) == 1:
        print(Ans[0]);
    if len(Ans) > 1:
        print("Bad magician!")
        
