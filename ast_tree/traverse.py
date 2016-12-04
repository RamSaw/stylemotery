import ast


def set_condition(mode):
    if mode == "parents":
        return lambda node: True if len(list(children(node))) > 0 else False
    elif mode == "leaves":
        return lambda node: True if len(list(children(node))) == 0 else False
    elif mode == "all":
        return lambda node: True


def bfs(node, callback, mode="all", out=None):
    condition = set_condition(mode)
    depth = 0

    def bfs_rec(node, depth, out):
        if condition(node):
            callback(node, depth, out)
        for child in children(node):
            bfs_rec(child, depth + 1, out)

    bfs_rec(node, depth, out)
    return out


def dfs(node, callback, mode="all", out=None):
    condition = set_condition(mode)
    depth = 0

    def dfs_rec(node, depth, out):
        for child in children(node):
            dfs_rec(child, depth + 1, out)
        if condition(node):
            callback(node, depth, out)

    dfs_rec(node, depth, out)
    return out


def children(node):
    if hasattr(node,"children"):
        for child in node.children:
            yield child
    else:
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                yield value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        yield item

def tree_print(tree,callback):
    bfs(tree, callback=callback, mode="all")

if __name__ == "__main__":
    pass
    # filename = os.path.join(os.getcwd(), 'dump_program.py')
    # traverse = bfs(ast_parse_file(filename), callback=printcb, mode="all")
    # astnodes = AstNodes()
    # basefolder = get_basefolder()
    # X, y, problems = parse_src_files(basefolder)
    # subX, subY, subProblems = split_tees(X, y, problems)
    #
    # print("\t\t%s Unique problems, %s Unique users :" % (len(set(problems)), len(set(y))))
    # print("\t\t%s All problems, %s All users :" % (len(problems), len(y)))
    # print("\t\t%s Sub problems, %s sub users :" % (len(subProblems), len(subY)))
    # ratio = sorted([(i, Counter(subY)[i],
    #                  "%{0}".format(round((Counter(subY)[i] / float(len(subY)) * 100.0), 2))) for i in Counter(subY)],
    #                key=itemgetter(1), reverse=True)
    # print("\t\t all users ratio ", ratio)
    # first_layer = []
    # for x in subX:
    #     first_layer.append(type(x).__name__)
    #
    # ratio = sorted([(i, Counter(first_layer)[i],
    #                  "%{0}".format(round((Counter(first_layer)[i] / float(len(first_layer)) * 100.0), 2))) for i in
    #                 Counter(first_layer)], key=itemgetter(1), reverse=True)
    # print("\t\t all users ratio ", ratio)

