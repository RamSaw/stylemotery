


import os
import random
import numpy as np

src_folder = "../train"
dst_folder = "../train/new_train"

classes = ['AMV0', 'Argaen0', 'Bastiandantilus0', 'Binnie0', 'BlackEagle0', 'ChevalierMalFet0', 'Coconut Big0', 'Eko0', 'EnTerr0', 'Entropy0', 'Fizu0', 'GauravRai0', 'Greatlemer0', 'IdahoJacket0', 'IdoLivneh0', 'J3ffreySmith0', 'Michael0', 'NaN0', 'Nooodles0', 'Phayr0', 'RalfKistner0', 'SickMath0', 'ToR0', 'YOBA0', 'addie90000', 'alexamici0', 'ana valeije0', 'anb0', 'aosuka0', 'bigOnion0', 'caethan0', 'cathco0', 'cheilman0', 'fractal0', 'gepa0', 'gizzywump0', 'graygrass0', 'hannanaha0', 'imakaramegane0', 'int n0', 'j4b0', 'jakab9220', 'jgaten0', 'joegunrok0', 'kawasaki0', 'kmod0', 'lookingfor0', 'max bublis0', 'mth0', 'netsuso0', 'nlse0', 'nwin0', 'oonishi0', 'pavlovic0', 'pawko0', 'pek0', 'pyronimous0', 'radkokotev0', 'rainmayecho0', 'raja baz0', 'rmmh0', 'ronnodas0', 'royf0', 'serialk0', 'shishkander0', 'taichino0', 'tama eguchi0', 'xoxie0', 'yordanmiladinov0', 'ziyan0']

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

authors = 55
prev_classes = set(classes[:55])
for i in range(5):
    with open(os.path.join(dst_folder, "55_authors.labels{0}.txt".format(i+1)), "+w") as wfile:
        wfile.write("Seed : {0}\n".format(str(random.randint(0,4294967295))))
        while True:
            np.random.shuffle(classes)
            new_classes = classes[:55]
            if prev_classes != set(new_classes):
                break
        wfile.write("Classes : {0}\n".format(str(new_classes)))
        prev_classes = new_classes
