from ast_tree.tree_nodes import AstNodes
from models.lstm_models import RecursiveLSTM, RecursiveBiLSTM
from models.tree_models import RecursiveTreeLSTM


def read_config(filename):
    with open(filename) as file:
        last_epoch = 1
        last_acc = 0
        params = 0
        yes = False
        for line in file:
            if "BatchNormalization" in line:
                yes =True
            if line.lower().startswith("params"):
                args_line = line.split(":",1)
                params = args_line[1].strip()
            elif line[0].isdigit() and "saved" in line:
                last_epoch = int(line.split()[0])+1
                last_acc = line.split()[-2]
            # elif line.startswith("Train labels "):
            #     train_lables = [v for idx, v in eval(line.split(":")[1])]
            # elif line.startswith("Test labels "):
            #     test_trees = [v for idx, v in eval(line.split(":")[1])]
        return params,last_epoch,last_acc,yes




import os
def getParams(layers, cell, units, authors):
    if cell == "lstm":
        model = RecursiveLSTM(units, authors, layers=layers, dropout=0.2,feature_dict=AstNodes(), classes=None, cell="lstm",residual=False)
    elif cell == "bilstm":
        model = RecursiveBiLSTM(units, authors, layers=layers, dropout=0.2,feature_dict=AstNodes(), classes=None,cell="lstm",residual=False)
    return model.params_count()


if __name__ == "__main__":
    basefolder = R"C:\Users\bms\Files\current\research\stylemotry\stylometry papers\best results\RNN\all"
    # folders = ['100','250','500']
    # for folder in folders:
    #     print(folder)
    files = [f for f in os.listdir(os.path.join(basefolder)) if f.endswith(".txt") and not f.startswith("all")]
    for file in files:
        params, last_epoch, last_acc,yes = read_config(os.path.join(basefolder,file))
        parts = file.split("_")
        layer, cell,units, authors = parts[0],parts[1],parts[2],parts[4]
        params = getParams(int(layer), cell,int(units), int(authors))
        print("\t{0:<2}{1:<10}{2:<10}{3:<4}{4:<10}{5:<20.10f}{6:<40}".format(layer,cell,units,authors,"{:,}".format(params),float(last_acc),file))
    print()