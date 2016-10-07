import os

from utils.extract_loss_progress import parse_result_file
from utils.graph import plot_each


def standarize(values):
    return {k.replace("_", " ").replace("training", "train"): v for k, v in values.items()}

def plot_results(base_folder,recursive=False):
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

    plot_each(new_results, base_folder)


if __name__ == "__main__":
    base_folder = R"C:\Users\bms\Files\study\DC\Experiments\results\best results\dropout0.2"
    plot_results(base_folder,recursive=False)
