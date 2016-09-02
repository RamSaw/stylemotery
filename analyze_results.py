




import os

from utils.extract_loss_progress import parse_result_file
from utils.graph import plot_all, plot_each

if __name__ == "__main__":
    base_folder = R"C:\Users\bms\Desktop\lstm\dropout"
    results = {}
    for filename in os.listdir(base_folder):
        if "results" in filename:
            loss = parse_result_file(os.path.join(base_folder,filename))
            results.update(loss)
    print(len(results))
    print(results.keys())
    for name,values in results.items():
        print(name," ==> ",values.keys())
    plot_each(results,base_folder)
