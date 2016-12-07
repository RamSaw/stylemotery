import os
import matplotlib.pyplot as plt
from utils.extract_loss_progress import parse_result_file
from utils.graph import plot_each, plot_each_smooth


def standarize(values):
    return {k.replace("_", " ").replace("training", "train"): v for k, v in values.items()}

def plot_results(base_folder,recursive=False,smooth=False):
    results = {}
    for filename in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, filename)) and recursive:
            plot_results(os.path.join(base_folder, filename))
        elif filename.endswith("_results.txt"):
            loss = parse_result_file(os.path.join(base_folder, filename))
            if len(next(iter(loss.values()))) > 0:
                results.update(loss)
    # print(results.keys())
    new_results = {name: standarize(values) for name, values in results.items()}
    for name, values in new_results.items():
        print(name, " ==> ", values.keys())

    # plot_each(new_results, base_folder)
    if smooth:
        plot_each_smooth(new_results, base_folder)

def extract_best_results(base_folder,recursive=False):
    results = {}
    for filename in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, filename)) and recursive:
            extract_best_results(os.path.join(base_folder, filename))
        elif filename.endswith("_results.txt"):
            loss = parse_result_file(os.path.join(base_folder, filename))
            if len(next(iter(loss.values()))) > 0:
                results.update(loss)
    # print(results.keys())
    new_results = {name: standarize(values) for name, values in results.items()}
    for name, values in new_results.items():
        print(name, " ==> ", values.keys())

    #plot_each(new_results, base_folder)
    for name, values in sorted(new_results.items()):
        print(name,end="\t")
        print("\t".join(name.split("_")),end="\t")
        saved = values["saved"]
        epoch = values["epoch"][saved[-1]]
        params = values["meta"]["Params"]
        test_accuracy = values["test accuracy"][saved[-1]]
        print("{0}\t{1}\t{2}".format(params,epoch,test_accuracy))

def extract_training_data(base_folder,recursive=False):
    results = {}
    for filename in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, filename)) and recursive:
            extract_training_data(os.path.join(base_folder, filename))
        elif filename.endswith("_results.txt"):
            loss = parse_result_file(os.path.join(base_folder, filename))
            if len(next(iter(loss.values()))) > 0:
                results.update(loss)
    # print(results.keys())
    new_results = {name: standarize(values) for name, values in results.items()}
    for name, values in new_results.items():
        print(name, " ==> ", values.keys())

    #plot_each(new_results, base_folder)
    for name, values in new_results.items():
        print(name)
        for k,v in values["meta"].items():
            if k in ["Seed","Classes"]:
                print("\t",k,":",v)

if __name__ == "__main__":
    base_folder = R"C:\Users\bms\Files\current\research\stylemotry\Experiments\best results\RNN"
    plot_results(base_folder,recursive=False,smooth=True)
