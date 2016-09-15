




import os

from utils.extract_loss_progress import parse_result_file
from utils.graph import plot_all, plot_each


def standarize(values):
    return {k.replace("_", " ").replace("training","train"): v for k, v in values.items()}


if __name__ == "__main__":
    base_folder = R"C:\Users\bms\Desktop\lstm\server1\dropout1"
    results = {}
    for filename in os.listdir(base_folder):
        if "results" in filename:
            loss = parse_result_file(os.path.join(base_folder,filename))
            results.update(loss)
    print(len(results))
    print(results.keys())
    new_results = {name:standarize(values) for name,values in results.items()}
    for name,values in new_results.items():
        print(name," ==> ",values.keys())

    plot_each(new_results,base_folder)
