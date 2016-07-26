"""
Google Code Jam 2012 Problem B
Usage:
    python problem_b.py < input.txt > output.txt
"""
import itertools
import sys


def calc_possible(n):
    combinations = itertools.combinations_with_replacement(list(range(n)), 3)  # 3 scores
    return filter(lambda x_y_z: x_y_z[0] + x_y_z[1] + x_y_z[2] == n, combinations)


def calc_surprising(n):
    results = list(filter(lambda scores: max(scores) - min(scores) == 2, calc_possible(n)))
    return results[0] if results else None


def calc_normal(n):
    results = list(filter(lambda scores: max(scores) - min(scores) <= 1, calc_possible(n)))
    return results[0] if results else None


def solve_problem():
    number_of_cases = int(sys.stdin.readline())

    for i in range(1, number_of_cases + 1):

        case = sys.stdin.readline().strip()
        result = 0
        num_of_googlers, num_of_surprising, desired_score, scores = case.split(' ', 3)
        num_of_googlers = int(num_of_googlers)
        num_of_surprising = int(num_of_surprising)
        desired_score = int(desired_score)
        scores = list(map(int, scores.split()))

        possible_scores = []

        for k in range(num_of_googlers):

            normal = calc_normal(scores[k])
            surprising = calc_surprising(scores[k])

            possible_scores.append(((normal if normal else (0, 0, 0), 0), (surprising if surprising else (0, 0, 0), 1)))

        possible = list(filter(lambda scores: sum([x[1] for x in scores]) == num_of_surprising, itertools.product(*possible_scores)))
        result = max([sum([int(max(x[0]) >= desired_score) for x in scores]) if scores else 0 for scores in possible])

        sys.stdout.write('Case #{0}: {1}\n'.format(i, result))


if __name__ == '__main__':
    solve_problem()
