import ast
import os
import platform

import numpy as np


def get_basefolder():
    if platform.system().startswith("Linux"):
        return R"/home/bms/projects/stylometory/stylemotery/dataset700"
    elif platform.system().startswith("Win"):
        return R"C:\Users\bms\PycharmProjects\stylemotery_code\dataset700"


def ast_parse_file(filename):
    try:
        with open(filename, 'r', encoding="utf-8") as file:
            return ast.parse(file.read())
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
    return np.array([ast_parse_file(name) for name in X_names]), y, problems

