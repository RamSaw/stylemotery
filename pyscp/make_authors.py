


import os
import random
import numpy as np

from utils.dataset_utils import parse_src_files

src_folder = "../train"
dst_folder = "../train/cpp"

# authors = [55]
# for filename in os.listdir(src_folder):
#     if os.path.isdir(os.path.join(src_folder, filename)):
#         continue
#     with open(os.path.join(src_folder,filename)) as file:
#         print(filename)
#         lines = [line.replace(":-",":") for line in file.readlines()]
#         classes = [v for idx, v in eval(lines[1].split(":")[1])]
#         with open(os.path.join(dst_folder,filename),"+w") as wfile:
#             wfile.write("Seed : "+str(random.randint()))
#             wfile.write("Classes : {0}".format(str(classes)))

authors = [5,25]
X, y, tags,features = parse_src_files(os.path.join("..","dataset","cpp"),seperate_trees=False,verbose=0)
classes = np.unique(y)
for author in authors:
    for i in range(5):
        with open(os.path.join(dst_folder, "{0}_authors.labels{1}.txt".format(author,i+1)), "+w") as wfile:
            wfile.write("Seed : {0}\n".format(str(random.randint(0,4294967295))))
            np.random.shuffle(classes)
            wfile.write("Classes : [{0}]\n".format(','.join(["'%s'" % c for c in  classes[:author]])))
            wfile.close()
