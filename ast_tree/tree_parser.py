import ast
import os
from collections import defaultdict
from ast_tree.tree_nodes import Node, print_dot_node
from ast_tree.traverse import tree_print, bfs, children
import sys
def ast_parse_file(filename):
    try:
        with open(filename, 'r', encoding="utf-8") as file:
            tree = ast.parse(file.read())
            return tree
    except Exception as e:
        print("ERROR: ", e, " filename", filename)

def parse_tree(filename):
    nodes = {}
    links = {}
    for line in open(filename):
        if line.startswith("<"):
            parts = line[1:].strip("\n").split("=")
            links[parts[0]] = parts[1].split(",")
        elif line.startswith(">"):
            parts = line[1:].strip("\n").split("\t")
            nodes[parts[0]] = Node(parts[1], parts[2], [])
    root_nodes = []
    for id, value in sorted(links.items()):
        for link in value:
            nodes[id].children.append(nodes[link])
            root_nodes.append(link)

    root_nodes = set(nodes.keys()) - set(root_nodes)
    root = Node("Program","",[nodes[id] for id in list(sorted(root_nodes))])
    return root

def parse_dot(filename):
    split = "\t"
    structs = []
    for line in open(filename):
        if line.startswith("//"):
            continue
        if line.startswith("strict graph"):
            structs.append(line)
        else:
            structs[-1] += line

    lines = []
    for struct in structs:
        s = struct.split("\n")
        for line in s:
            if line.startswith("\t"):
                lines.append(line.strip().replace("\n", split))
            else:
                if len(lines) > 0:
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
                        links[int(numbers[0].strip())].append(int(numbers[1].strip()))
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
    root_nodes = []
    for id, value in sorted(links.items()):
        for link in value:
            nodes[id].children.append(nodes[link])
            root_nodes.append(link)

    root_nodes = set(nodes.keys()) - set(root_nodes)
    root = Node("Program","",[nodes[id] for id in list(sorted(root_nodes))])
    return root

def fast_parse_dot(filename):
    split = "\t"
    lines = []
    for line in open(filename):
        if line.startswith("\\"):
            continue
        if line.startswith("\t"):
            lines.append([line.strip().replace("\n", split)])
        else:
            if len(lines) > 0:
                lines[-1].append(line.replace("\n", split).strip())
    nodes = {}
    links = defaultdict(list)
    for line in lines:
        if len(line) > 0 and line[0][0].isdigit():
            try:
                if "--" in line[0][:10]:
                    import re
                    non_decimal = re.compile(R"[^\d-]+")
                    numbers = non_decimal.sub('', line[0]).split("--")
                    try:
                        links[int(numbers[0].strip())].append(int(numbers[1].strip()))
                    except Exception as e1:
                        print(e1)
                # int(parts[0])
                else:
                    if len(line) > 1:
                        content = {"type": "", "code": ""}
                        for l in line:
                            if "code:" in l:
                                content["code"] = l.split(":")[1]
                            elif "type:" in l:
                                content["type"] = l.split(":")[1]
                        id = int(line[0].split("\t")[0])
                        nodes[id] = Node(content["type"], content["code"], [])
            except Exception as e:
                print(e)
    root_nodes = []
    for id, value in sorted(links.items()):
        for link in value:
            nodes[id].children.append(nodes[link])
            root_nodes.append(link)

    root_nodes = set(nodes.keys()) - set(root_nodes)
    root = Node("Program","",[nodes[id] for id in list(sorted(root_nodes))])
    return root