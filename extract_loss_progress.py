from collections import defaultdict


def parse_loss_file(filename):
    loss_history = defaultdict(list)
    epoch = 0
    test = False
    with open(filename) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Epoch:"):
                epoch = line[line.index(':'):line.index("/")].strip()
                test = False
            elif "[" in line and not test:
                iteration = line[:8].split("/")[0].strip()
                loss = line[line.rindex(":")+1:].replace("\b","").strip()
                loss_history[epoch].append((iteration,loss))
            elif "Test evaluateion" in line:
                test = True
    return loss_history





import os


def extract_loss(history):
    loss = []
    for key,values in sorted(history.items()):
        epoch_vals,loss_vals = zip(values)
        max_loss = max(loss_vals)
        loss.append(max_loss)
    return loss



if __name__ == "__main__":
    filenames = ["1_lstm_500_dropout_batch_alldataset_cliping"]
    basefolder = R"C:\Users\bms\Desktop\lstm\fulldataset"
    for filename in [os.path.join(basefolder,x,".txt") for x in filenames]:
        history = parse_loss_file(filename)
        loss = extract_loss(history)
        with open(os.path.join(basefolder,filename,"_parsed",".txt"),mode="+w") as wfile:
            wfile.write("\t".join(loss))

