import ast
import os
import platform
import sys
import numpy as np

def get_basefolder():
    if platform.system().startswith("Linux"):
        return R"dataset700"
    elif platform.system().startswith("Win"):
        return R"C:\Users\bms\PycharmProjects\stylemotery_code\dataset700"


def ast_parse_file(filename):
    try:
        with open(filename, 'r', encoding="utf-8") as file:
            tree = ast.parse(file.read())
            return tree
    except Exception as e:
        print("ERROR: ", e, " filename", filename)


def get_src_files(basefolder):
    files = os.listdir(basefolder)
    files_noext = ['.'.join(s.split('.')[:-1]) for s in files]

    problems = [p.split('.')[0] for p in files_noext]
    users = [' '.join(p.split('.')[1:]) for p in files_noext]

    return np.array([os.path.join(basefolder, file) for file in files]), np.array(users), np.array(problems)


def parse_src_files(basefolder, seperate_trees=False):
    X_names, y, problems = get_src_files(basefolder)
    return np.array([ast_parse_file(name) for name in X_names]), np.array(y), problems


def generate_tree(node, children):
    root = ast.Module()
    root._fields = ("body",)

    ast_nodes = [ast.Add, ast.Assign,ast.And,ast.arguments,ast.AugAssign]

    child = node()
    child._fields = ("body",)
    child.body = [ast_nodes[i]() for i in range(children)]

    root.body = [child for i in range(children)]
    return root


def generate_trees(basefolder, labels=2, children=10, examples_per_label=10):
    X = []
    y = []
    ast_nodes = [ast.Add, ast.Assign]
    for label in range(labels):
        tree = generate_tree(ast_nodes[label], children)
        X.extend([tree for i in range(examples_per_label)])
        y.extend([label for i in range(examples_per_label)])
    return np.array(X), np.array(y), np.array([])


def make_backward_graph(basefolder, filename, var):
    import chainer.computational_graph as c
    import os
    g = c.build_computational_graph(var)
    with open(os.path.join(basefolder, filename), '+w') as o:
        o.write(g.dump())
    # dot -Tps filename.dot -o outfile.ps
    from subprocess import call
    call(["dot", "-Tpdf", os.path.join(basefolder, filename), "-o", os.path.join(basefolder, filename + ".pdf")])


def print_model(model,depth=0,output=sys.stdout):
    if len(list(model.children())) == 0:
        output.write("\t"*depth+"{0} {1} {2} \n".format(model.name,type(model).__name__,model._params))
        for p in model._params:
            output.write("\t"*depth+"{0} {1} {2} {3}\n".format("\t"*depth, p, "=>", getattr(model, p).data.shape))
    else:
        if model.name is not None:
            output.write("\t"*depth+"{0} \n".format(model.name))
        for child in model.children():
            print_model(child,depth=depth+1,output=output)


