import ast
import os

from ast_tree.traverse import tree_print
from ast_tree.tree_nodes import print_ast_node, print_dot_node
from ast_tree.tree_parser import ast_parse_file, parse_tree
from utils.analysis_utils import max_depth, max_branch

stack = []
def print_ast_node_tree(node, depth, out=None):
    global stack
    '''print indented node names'''
    nodename = "["+node.__class__.__name__
    # if hasattr(node,"_fields"):
    for i,(name, value) in enumerate(ast.iter_fields(node)):
        field = getattr(node, name)
        if type(field) in [int, str, float]:
            if i == 0:
                nodename += "("
            nodename += str(getattr(node, name))
    if "(" in nodename:
        nodename += ")"
    # elif hasattr(node,"_content"):
    #     for name, value in node.content():
    #         field = getattr(node, name)
    #         if type(field) in [int, str, float]:
    #             nodename += str(getattr(node, name))
    while len(stack) > 0 and stack[0] >= depth:
        print(' ' * stack[0] * 2 + "]")
        stack = stack[1:]
    stack = [depth] + stack
    print(' ' * depth * 2 + nodename)


if __name__ == "__main__":
    # filename = os.path.join("..","ast_tree", 'dump_program.py')
    # ast_tree = ast_parse_file(filename)
    # # print(list(ast_paths(ast_tree)))
    # tree_print(ast_tree,callback=print_ast_node_tree)
    # for s in stack:
    #     print(' ' * s * 2 + "]")

    filename = os.path.join("..","dataset","all_cpp",'Brian_Harris.50.p_saveg.tree')
    ast_tree = parse_tree(filename)
    # print(list(ast_paths(ast_tree)))
    tree_print(ast_tree[0],callback=print_dot_node)
    print(max_depth(ast_tree[0]))
    print(max_branch(ast_tree[0]))
    # for s in stack:
    #     print(' ' * s * 2 + "]")
