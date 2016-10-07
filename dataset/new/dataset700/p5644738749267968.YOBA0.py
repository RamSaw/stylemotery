import collections
import functools
import operator


def optimal_war_step(blocks, value):

    win_blocks = set([block for block in blocks if block > value])
    blocks.discard(min(win_blocks if win_blocks else blocks))
    return bool(win_blocks)


def deceitful_war(ken, naomi, epsilon = 10 ** (-6)):

    while ken:

        min_ken = min(ken)
        max_ken = max(ken)
        min_naomi = min(naomi)

        if min_ken > min_naomi and any([operator.lt(*bs) for bs in zip(sorted(naomi), sorted(ken))]):

            yield max_ken - epsilon

        else:

            yield 1.0 - epsilon

        naomi.discard(min_naomi)


for i in range(int(eval(input()))):

    eval(input())
    naomi = set(map(float, str.split(eval(input()))))
    ken = set(map(float, str.split(eval(input()))))

    dwar_ken = ken.copy()
    dwar_naomi = naomi.copy()
    dwar = collections.Counter(list(map(functools.partial(optimal_war_step, dwar_ken), deceitful_war(dwar_ken, dwar_naomi))))
    war = collections.Counter(list(map(functools.partial(optimal_war_step, ken), naomi)))
    print((str.format("Case #{}: {} {}", i + 1, dwar[False], war[False])))
