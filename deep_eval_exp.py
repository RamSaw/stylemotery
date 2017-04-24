import os
import random

import chainer
from chainer import optimizers
# from deep_ast.tree_lstm.treelstm import TreeLSTM
from chainer import serializers
from models.lstm_models import RecursiveLSTM, RecursiveBiLSTM
from models.tree_models import RecursiveTreeLSTM
from utils.exp_utlis import split_trees, pick_subsets, evaluate, train, evaluate_relax, read_train_config, \
    evaluate_ensemble, split_trees2
from utils.dataset_utils import parse_src_files, print_model
import argparse
import sys
from argparse import Namespace


def print_table(table):
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print("| " + " | ".join("{:{}}".format(x, col_width[i])
                                for i, x in enumerate(line)) + " |")


def read_config(filename):
    with open(filename) as file:
        last_epoch = 1
        saved_epochs = []
        for line in file:
            if line.startswith("Args "):
                args_line = line.split(":-", 1)
                # line = "["+args_line[1].strip().replace("Namespace(","")[:-1]+"]"
                args = eval(args_line[1])
            elif line.startswith("Seed "):
                args_line = line.split(":-", 1)
                seed = int(args_line[1].strip())
            elif line.startswith("Classes "):
                classes = [v for idx, v in eval(line.split(":-")[1])]
            elif line[0].isdigit():
                last_epoch = int(line[0]) + 1
                if "saved" in line:
                    saved_epochs.append(line.split()[0])
                # elif line.startswith("Train labels "):
                #     train_lables = [v for idx, v in eval(line.split(":")[1])]
                # elif line.startswith("Test labels "):
                #     test_trees = [v for idx, v in eval(line.split(":")[1])]
        return args, seed, classes, saved_epochs, last_epoch


def remove_old_model(models_base_folder, exper_name, epoch_):
    model_saved_name = "{0}_epoch_{1}.my".format(exper_name, epoch_)
    path = os.path.join(models_base_folder, model_saved_name)
    if os.path.exists(path):
        os.remove(path)
    model_saved_name = "{0}_epoch_{1}.opt".format(exper_name, epoch_)
    path = os.path.join(models_base_folder, model_saved_name)
    if os.path.exists(path):
        os.remove(path)


def save_new_model(model, optimizer, models_base_folder, exper_name, epoch):
    model_saved_name = "{0}_epoch_{1}.my".format(exper_name, epoch)
    path = os.path.join(models_base_folder, model_saved_name)
    serializers.save_npz(path, model)
    # save optimizer
    model_saved_name = "{0}_epoch_{1}.opt".format(exper_name, epoch)
    path = os.path.join(models_base_folder, model_saved_name)
    serializers.save_npz(path, optimizer)


def main_experiment(ensembles):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default="default_experiment", help='Experiment name')
    parser.add_argument('--config', '-c', type=str, default="", help='Configuration file')

    args = parser.parse_args()
    if args.config == "":
        parser.print_help()
        return
    args, seed, classes, saved_epochs,last_epoch = read_config(args.config)

    n_units = args.units
    batch_size = args.batchsize
    gpu = args.gpu
    models_base_folder = "saved_models"
    output_folder = os.path.join("results",args.folder)  # args.folder  #R"C:\Users\bms\PycharmProjects\stylemotery_code" #
    exper_name = args.name
    dataset_folder = os.path.join("dataset", args.dataset)
    seperate_trees = args.seperate
    model_name = args.model
    layers = args.layers
    dropout = args.dropout
    cell = "lstm"#args.cell
    residual = args.residual

    output_file = sys.stdout#open(os.path.join(output_folder, exper_name + "_results.txt"), mode="a")

    trees, tree_labels, lable_problems, tree_nodes = parse_src_files(dataset_folder, seperate_trees=seperate_trees)
    if args.train:
        rand_seed, classes = read_train_config(os.path.join("train", args.dataset, args.train))
        trees, tree_labels = pick_subsets(trees, tree_labels, classes=classes)
    else:
        if args.classes > -1:
            trees, tree_labels = pick_subsets(trees, tree_labels, labels=args.classes, seed=rand_seed, classes=None)

    if model_name == "lstm":
        model = RecursiveLSTM(n_units, len(classes), layers=layers, dropout=dropout, feature_dict=tree_nodes,
                              classes=classes, cell=cell, residual=residual)
    elif model_name == "bilstm":
        model = RecursiveBiLSTM(n_units, len(classes), layers=layers, dropout=dropout, feature_dict=tree_nodes,
                                classes=classes, cell=cell, residual=residual)
    elif model_name == "treelstm":
        model = RecursiveTreeLSTM(n_children=layers, n_units=n_units, n_label=len(classes), dropout=dropout,
                                  feature_dict=tree_nodes, classes=classes)
    else:
        print("No model was found")
        return

    models = []

    # for epoch in saved_epochs[::1][:ensembles]:
    # # load the model
    #     model_saved_name = "{0}_epoch_{1}".format(exper_name,epoch)
    #     output_file.write("load {0} ... \n".format(model_saved_name))
    #     saved_models = [m for m in os.listdir(os.path.join(models_base_folder,"lstm2"))
    #                     if m == model_saved_name + ".my"]
    #     if len(saved_models) > 0:
    #         # pick the best one
    #         model_saved_name = list(sorted(saved_models, key=lambda name: int(name.split(".")[0].split("_")[-1]), reverse=True))[0]
    #     else:
    #         print("No model was found to load")
    #         return
    #     path = os.path.join(models_base_folder,"lstm2",model_saved_name)
    #     serializers.load_npz(path, model)
    #     # if gpu >= 0:
    #     #     model.to_gpu()
    #     models.append(model)

    model_files = [f for f in os.listdir(os.path.join(models_base_folder,"bilstm")) if f.endswith(".my")]
    for model_saved_name in model_files:
    # load the model
    #     if exper_name in model_saved_name:
        output_file.write("load {0} ... \n".format(model_saved_name))
        path = os.path.join(models_base_folder,"bilstm",model_saved_name)
        serializers.load_npz(path, model)
        # if gpu >= 0:
        #     model.to_gpu()
        models.append(model)



    # trees, tree_labels = pick_subsets(trees, tree_labels, classes=classes)
    train_trees, train_lables, test_trees, test_lables, classes, cv = split_trees(trees, tree_labels, n_folds=5,
                                                                                  shuffle=True, seed=seed,
                                                                                  iterations=args.iterations)
    # print('Train')
    # output_file.write("Test  labels :- (%s,%s%%): %s\n" % (len(test_lables), (len(test_lables) / len(tree_labels)) * 100, test_lables))

    # output_file.write("{0:<10}{1:<20}\n".format("Relax", "test_accuracy"))
    # print('Relax evaluation: ')
    # for i in [1, 5, 10, 15]:
    #     test_accuracy, test_loss = evaluate_relax(model, test_trees, test_lables, batch_size=batch_size, progbar=True, relax=i)
    #     # test_accuracy, test_loss = evaluate(model, test_trees, test_lables, batch_size=batch_size)
    #     print()
    #     output_file.write("{0:<10}{1:<20.10f}\n".format(i,test_accuracy))
    #     output_file.flush()

    # print("One model:")
    # import numpy as np
    # classes_num = np.arange(len(classes))
    # test_accuracy, test_loss = evaluate(models[0], test_trees, test_lables,classes=classes_num, batch_size=batch_size)
    # test_accuracy, test_loss = evaluate(model, test_trees, test_lables, batch_size=batch_size)
    # output_file.write("{0:<20.10f}\n".format(test_accuracy))
    # output_file.flush()

    print("Ensmbel:")
    test_accuracy, test_loss = evaluate_ensemble(models, test_trees, test_lables, batch_size=batch_size)
    # test_accuracy, test_loss = evaluate(model, test_trees, test_lables, batch_size=batch_size)
    output_file.write("{0:<20.10f}\n".format(test_accuracy))
    output_file.flush()

    output_file.close()
if __name__ == "__main__":
    main_experiment(5)
