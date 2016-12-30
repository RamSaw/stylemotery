import os
import csv
import shutil
from collections import defaultdict

def filter(src,dst,limit):
    labels = defaultdict(list)
    rows = []
    with open(src) as ofile:
        reader = csv.reader(ofile, delimiter='\t')
        header = [h.strip() for h in next(reader)]
        for id, line in enumerate(reader):
            cols = [s.strip() for s in line]
            if cols[1] == "0.BitSet.tree": #Dennis_Luxen.
                pass
            try:
                avg_d = int(cols[2])
                max_b = int(cols[5])
                if avg_d >= 4 and 7 < max_b <= 200:
                    rows.append(cols)
                    labels[cols[0]].append(len(rows)-1)
            except Exception:
                print(cols)
                return

    selected_rows = []
    selected_labels = []
    for label, indices in sorted(labels.items()):
        indices = indices[:limit]
        if len(indices) >= limit-1:
            selected_labels.append(label)
            print(label, " => ", len(indices))
            for id in indices:
                selected_rows.append(rows[id])

    with open(dst, "w+") as wfile:
        wfile.write("\t".join(header) + "\n")
        for row in selected_rows:
            wfile.write("\t".join(row) + "\n")
    print("final labels=", len(labels))

def extract_cpp_dataset(srcfile,basefolder,dstfolder):
    with open(srcfile) as ofile:
        reader = csv.reader(ofile, delimiter='\t')
        header = [h.strip() for h in next(reader)]
        for id, line in enumerate(reader):
            cols = [s.strip() for s in line]
            shutil.copy(os.path.join(basefolder,cols[0]+"."+cols[1]),
                        os.path.join(dstfolder, cols[0] + "." + cols[1]))

if __name__ == "__main__":
    srcfile = R"C:\Users\bms\Files\current\research\stylemotry\stylemotery_code\dataset\output.csv"
    basefolder = R"C:\Users\bms\Files\current\research\stylemotry\stylemotery_code\dataset\dcpp"
    dstfolder = R"C:\Users\bms\Files\current\research\stylemotry\stylemotery_code\dataset\fdcpp"
    filter(R"C:\Users\bms\Files\current\research\stylemotry\stylemotery_code\dataset\analysis\text.txt",srcfile,20)
    extract_cpp_dataset(srcfile,basefolder,dstfolder)

