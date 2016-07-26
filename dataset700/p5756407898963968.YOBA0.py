def read_case():

    answer = int(eval(input()))
    lines = tuple([set(str.split(eval(input()))) for _ in range(4)])
    return lines[answer - 1]


for i in range(int(eval(input()))):

    intersection = read_case() & read_case()
    count = len(intersection)
    if count == 1:

        answer = intersection.pop()

    elif count > 1:

        answer = "Bad magician!"

    elif count < 1:

        answer = "Volunteer cheated!"

    print((str.format("Case #{}: {}", i + 1, answer)))
