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


def parse_result_file(filename):
    header = defaultdict(str)
    loss_history = defaultdict(list)
    epoch_idx = 0
    model_name = ""
    with open(filename) as file:
        lines = file.readlines()
        for i,line in enumerate(lines):
            line = line.replace(":-",":")
            if ":" in line:
                meta = line.split(":",1)
                header[meta[0].strip()] = meta[1].strip()
            if line.strip().startswith("Model:"):
                model_name = line.split(":",1)[1].strip()
            if line.strip().startswith("Evaluation"):
                epoch_idx = i
                break
        epoch_idx += 1
        list_names = [name.strip() for name in lines[epoch_idx].strip().split("\t")]
        if len(list_names) < 2:
            list_names = [name.strip() for name in lines[epoch_idx].strip().split()]
        list_names += ["saved"]
        for epoch, line in enumerate(lines[epoch_idx+1:]):
            if "stopping" not in line.lower():
                values = line.strip().split()
                for i,name in enumerate(list_names[:-1]):
                    if i < len(values) and i < len(list_names):
                        loss_history[list_names[i]].append(values[i])
                if values[-1] == "saved":
                    loss_history[list_names[-1]].append(epoch)
        loss_history["meta"] = header
    return {model_name:loss_history}


import os
def extract_loss(history):
    loss = []
    for key,values in sorted(history.items()):
        loss_vals = [l for e,l in values]
        max_loss = max(loss_vals)
        loss.append(max_loss)
    return loss



if __name__ == "__main__":
    # filenames = ["2_lstm_500_dropout_batch_alldataset_cliping"]
    # basefolder = R"C:\Users\bms\Desktop\lstm\partialdatasets"
    # for filename in [os.path.join(basefolder,x+".txt") for x in filenames]:
    #     history = parse_loss_file(filename)
    #     loss = extract_loss(history)
    #     with open(os.path.join(basefolder,filename+"_parsed.txt"),mode="+w") as wfile:
    #         wfile.write("\t".join(loss))
    loss = parse_result_file(os.path.join(R"C:\Users\bms\PycharmProjects\stylemotery_code\out","1_lstm_dropout_500_2_labels1_results.txt"))

