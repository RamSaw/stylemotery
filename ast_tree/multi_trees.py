from ast_tree.traverse import children
from copy import deepcopy
import numpy as np

def split_trees(X, y, problems):
    subX = []
    subY = []
    subProblem = []
    for i, tree in enumerate(X):
        ast_children = children(tree)
        for child in ast_children:
            subX.append(child)
            subY.append(y[i])
            subProblem.append(problems[i])
    return np.array(subX), np.array(subY), np.array(subProblem)


def split_trees2(X, y, problems,original=False):

    subX = list(X) if original else []
    subY = list(y) if original else []
    subProblem = list(problems) if original else []

    for i, tree in enumerate(X):
        functions = []
        classes = []
        imports = []
        global_code = []
        ast_children = children(tree)
        for child in ast_children:
            child_name = type(child).__name__
            if child_name == "FunctionDef":
                functions.append(child)
            elif child_name == "ClassDef":
                classes.append(child)
            elif child_name == "ImportFrom" or child_name == "Import":
                imports.append(child)
            else:
                global_code.append(child)
        # add functions
        subX.extend(functions)
        subY.extend([y[i]] * len(functions))
        # add classes
        subX.extend(classes)
        subY.extend([y[i]] *len(classes))
        # add imports
        tree.body = []
        import_module = copy.deepcopy(tree)
        import_module.body = imports
        subX.append(import_module)
        subY.append(y[i])
        # add the rest of the code ( global instructions)
        tree.body = []
        global_module = copy.deepcopy(tree)
        global_module.body = global_code
        subX.append(global_module)
        subY.append(y[i])

        subProblem.append(problems[i])
    return np.array(subX), np.array(subY), np.array(subProblem)