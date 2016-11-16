from ast_tree.ast_parser import ast_print
from utils.fun_utils import parse_src_files, get_basefolder, make_binary_tree
import os

if __name__ == "__main__":
    trees, tree_labels, lable_problems = parse_src_files(os.path.join(".." ,get_basefolder()))
    ast_print(trees[0])
    binary_tree = make_binary_tree(trees[0])
    ast_print(binary_tree)