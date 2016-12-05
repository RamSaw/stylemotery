from collections import defaultdict
import os
import numpy as np
import shutil
from tqdm import tqdm

from ast_tree.tree_nodes import Node
from ast_tree.tree_parser import parse_tree

nodetypes = [s.upper() for s in ['AndExpression', 'ExclusiveOrExpression', 'IncDec', 'ParameterType', 'Parameter',
                                      'IdentifierDecl',
                                      'ContinueStatement', 'CompoundStatement', 'PrimaryExpression', 'Expression',
                                      'AdditiveExpression',
                                      'ExpressionStatement', 'DoStatement', 'SwitchStatement', 'CastExpression',
                                      'InclusiveOrExpression',
                                      'Label', 'IncDecOp', 'ClassDefStatement', 'Sizeof', 'MemberAccess',
                                      'EqualityExpression',
                                      'UnaryOperator', 'WhileStatement', 'ConditionalExpression', 'ParameterList',
                                      'CastTarget',
                                      'InitializerList', 'IfStatement', 'ElseStatement', 'RelationalExpression',
                                      'BlockStarter',
                                      'ReturnStatement', 'GotoStatement', 'UnaryExpression', 'ArrayIndexing',
                                      'ArgumentList',
                                      'ReturnType',
                                      'Statement', 'AssignmentExpr', 'OrExpression', 'FunctionDef', 'CallExpression',
                                      'IdentifierDeclStatement', 'PtrMemberAccess', 'UnaryOp',
                                      'MultiplicativeExpression', 'Argument',
                                      'BitAndExpression', 'ShiftExpression', 'Identifier', 'Condition', 'ForStatement',
                                      'Callee',
                                      'IdentifierDeclType', 'SizeofExpr', 'BreakStatement', 'ForInit', 'SizeofOperand',
                                      "Program"]]

def parse_dot2(src_file,dst_file):
    split = "\t"
    structs = []
    for line in src_file:
        if line.startswith("//"):
            continue
        if line.startswith("strict graph"):
            structs.append(line)
        else:
            structs[-1] += line

    lines = []
    noTab = False
    for struct in structs:
        s = struct.split("\n")
        for line in s:
            if line.startswith("\t"):
                lines.append(line.strip().replace("\n", split))
            else:
                if len(lines) > 0:
                    if line.startswith("type:\\"):
                        line = line[:5]
                        lines[-1] = lines[-1] + split + line.replace("\n", split).strip()
                        noTab = True
                    else:
                        if noTab:
                            lines[-1] = lines[-1] + line.replace("\n", split).strip()
                            noTab = False
                        else:
                            lines[-1] = lines[-1] + split + line.replace("\n", split).strip()
        lines.append("")
    nodes = {}
    links = defaultdict(list)
    for line in lines:
        if len(line) > 0 and line[0].isdigit():
            try:
                if "--" in line[:10]:
                    import re
                    non_decimal = re.compile(R"[^\d-]+")
                    line = non_decimal.sub('', line)
                    numbers = line.split("--")
                    try:
                        links[int(numbers[0].strip())].append(numbers[1].strip())
                    except Exception as e1:
                        print(e1)
                # int(parts[0])
                else:
                    parts = line.split("\t")
                    if len(parts) > 1:
                        content = {"type": "", "code": ""}
                        for part in parts[1:]:
                            if part.startswith("type:"):
                                content["type"] = part.split(":")[1]
                            if part.startswith("code:"):
                                content["code"] = part.split(":")[1]
                        if content['type'] == '':
                            type = line.split("type:")
                            if len(type) > 1:
                                content["type"] = type[1].replace("\\", "").split()[0]
                        nodes[int(parts[0])] = Node(content["type"], content["code"], [])
                        nodetypes.index(content['type'].upper())
            except Exception as e:
                print(e)
                print(src_file.name)
    for key,node in sorted(nodes.items()):
        dst_file.write(">{0}\t{1}\t{2}\n".format(key,node.type,node.code))
    for key,link in sorted(links.items()):
        dst_file.write("<{0}={1}\n".format(key,','.join(link)))

def parse_dot(src_file,dst_file):
    split = "\t"
    structs = []
    for line in src_file:
        if line.startswith("//"):
            continue
        if line.startswith("strict graph"):
            structs.append(line)
        else:
            structs[-1] += line

    lines = []
    noTab = False
    for struct in structs:
        s = struct.split("\n")
        for line in s:
            if line.startswith("\t"):
                lines.append(line.strip().replace("\n", split))
            else:
                if len(lines) > 0:
                    if line.startswith("type:\\"):
                        line = line[:5]
                        lines[-1] = lines[-1] + split + line.replace("\n", split).strip()
                        noTab = True
                    else:
                        if noTab:
                            lines[-1] = lines[-1] + line.replace("\n", split).strip()
                            noTab = False
                        else:
                            lines[-1] = lines[-1] + split + line.replace("\n", split).strip()
        lines.append("")
    nodes = {}
    links = defaultdict(list)
    for line in lines:
        if len(line) > 0 and line[0].isdigit():
            try:
                if "--" in line[:10]:
                    import re
                    non_decimal = re.compile(R"[^\d-]+")
                    line = non_decimal.sub('', line)
                    numbers = line.split("--")
                    try:
                        links[int(numbers[0].strip())].append(numbers[1].strip())
                    except Exception as e1:
                        print(e1)
                # int(parts[0])
                else:
                    parts = line.split("\t")
                    if len(parts) > 1:
                        content = {"type": "", "code": ""}
                        for part in parts[1:]:
                            if part.startswith("type:"):
                                content["type"] = part.split(":")[1]
                            if part.startswith("code:"):
                                content["code"] = part.split(":")[1]
                        nodes[int(parts[0])] = Node(content["type"], content["code"], [])
            except Exception as e:
                print(e)
    for key,node in sorted(nodes.items()):
        dst_file.write(">{0}\t{1}\t{2}\n".format(key,node.type,node.code))
    for key,link in sorted(links.items()):
        dst_file.write("<{0}={1}\n".format(key,','.join(link)))

def gen_tree_files(basefolder):
    trees = []
    users = []
    problems = []
    for folder in [f for f in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder,f))]:
        print(folder, " ...", end=" ", flush=True)
        for number in os.listdir(os.path.join(basefolder, folder)):
            file = [filename for filename in os.listdir(os.path.join(basefolder, folder, number)) if
                    filename.endswith(".dot")][0]
            src_file = open(os.path.join(basefolder, folder,number, file))
            dst_file = open(os.path.join(basefolder, folder,number, os.path.splitext(file)[0]+".tree"),"w+")
            parse_dot2(src_file,dst_file)
        print("done.")


def get_dot_files(basefolder):
    trees = []
    users = []
    problems = []
    for folder in [f for f in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder,f))]:
        for number in os.listdir(os.path.join(basefolder, folder)):
            file = [filename for filename in os.listdir(os.path.join(basefolder, folder, number)) if
                    filename.endswith(".dot")][0]
            trees.append(parse_dot(os.path.join(basefolder, folder, number, file)))
            users.append(folder)
    return np.array(trees),np.array(users),np.array(problems)


def traverse(basefolder):
    trees = []
    users = []
    problems = []
    for folder in [f for f in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder,f))]:
        print(folder, " ...", end=" ", flush=True)
        for number in os.listdir(os.path.join(basefolder, folder)):
            file = [filename for filename in os.listdir(os.path.join(basefolder, folder, number)) if
                    filename.endswith(".tree")][0]
            trees.append(parse_tree(os.path.join(basefolder, folder, number, file)))
            users.append(folder)
            print(number, end=" ", flush=True)
        print("done.")
    return np.array(trees),np.array(users),np.array(problems)

def traverse1(basefolder):
    trees = []
    users = []
    problems = []
    for file in tqdm(os.listdir(basefolder)):
        trees.append(parse_tree(os.path.join(basefolder, file)))
        users.append(file.split('.')[0])
    return np.array(trees),np.array(users),np.array(problems)

def copy_trees(basefolder,dstfolder):
    for folder in [f for f in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder,f))]:
        print(folder, " ...", end=" ", flush=True)
        for number in os.listdir(os.path.join(basefolder, folder)):
            file = [filename for filename in os.listdir(os.path.join(basefolder, folder, number)) if
                    filename.endswith(".tree")][0]
            shutil.copy(os.path.join(basefolder, folder, number, file),
                        os.path.join(dstfolder, folder+"."+number+"."+file ))
        print("done.")
basefolder = R"C:\Users\bms\Files\current\research\stylemotry\stylemotery_code\dataset\cpp"
dstfolder = R"C:\Users\bms\Files\current\research\stylemotry\datasets\balanced_cpp"
# copy_trees(dstfolder,basefolder)
traverse1(basefolder)
# file = R"C:\Users\bms\Files\current\research\stylemotry\datasets\33\components.dot"
# dfile = R"C:\Users\bms\Files\current\research\stylemotry\datasets\33\components.tree2"
# parse_dot2(open(file),open(dfile,"w+"))
# max_authors = 5
# print(max_authors,"*"*max_authors)
# for i in range(5):
#     random.shuffle(users)
#     print(','.join(["'{0}'".format(s) for s in users[:max_authors]]))
#
#
# max_authors = 15
# print(max_authors,"*"*max_authors)
# for i in range(5):
#     random.shuffle(users)
#     print(','.join(["'{0}'".format(s) for s in users[:max_authors]]))
#
# max_authors = 25
# print(max_authors,"*"*max_authors)
# for i in range(5):
#     random.shuffle(users)
#     print(','.join(["'{0}'".format(s) for s in users[:max_authors]]))



