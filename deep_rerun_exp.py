import os

import chainer
from chainer import optimizers
# from deep_ast.tree_lstm.treelstm import TreeLSTM
from chainer import serializers
from models.lstm_models import RecursiveLSTM, RecursiveBiLSTM
from models.tree_models import RecursiveTreeLSTM
from utils.exp_utlis import split_trees, pick_subsets, evaluate, train
from utils.dataset_utils import parse_src_files, print_model
import argparse
from argparse import Namespace

def print_table(table):
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print("| " + " | ".join("{:{}}".format(x, col_width[i])
                                for i, x in enumerate(line)) + " |")

def read_config(filename):
    with open(filename) as file:
        last_epoch = 1
        for line in file:
            if line.startswith("Args "):
                args_line = line.split(":-",1)
                # line = "["+args_line[1].strip().replace("Namespace(","")[:-1]+"]"
                args = eval(args_line[1])
            elif line.startswith("Seed "):
                args_line = line.split(":-",1)
                seed = int(args_line[1].strip())
            elif line.startswith("Classes "):
                classes = [v for idx, v in eval(line.split(":-")[1])]
            elif line[0].isdigit():
                last_epoch = int(line[0])+1
            # elif line.startswith("Train labels "):
            #     train_lables = [v for idx, v in eval(line.split(":")[1])]
            # elif line.startswith("Test labels "):
            #     test_trees = [v for idx, v in eval(line.split(":")[1])]
        return args,seed,classes,last_epoch

def remove_old_model(models_base_folder,exper_name, epoch_):
    model_saved_name = "{0}_epoch_{1}.my".format(exper_name, epoch_)
    path = os.path.join(models_base_folder, model_saved_name)
    if os.path.exists(path):
        os.remove(path)
    model_saved_name = "{0}_epoch_{1}.opt".format(exper_name, epoch_)
    path = os.path.join(models_base_folder, model_saved_name)
    if os.path.exists(path):
        os.remove(path)


def save_new_model(model,optimizer,models_base_folder,exper_name, epoch):
    model_saved_name = "{0}_epoch_{1}.my".format(exper_name, epoch)
    path = os.path.join(models_base_folder, model_saved_name)
    serializers.save_npz(path, model)
    # save optimizer
    model_saved_name = "{0}_epoch_{1}.opt".format(exper_name, epoch)
    path = os.path.join(models_base_folder, model_saved_name)
    serializers.save_npz(path, optimizer)

def main_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="", help='Configuration file')

    args = parser.parse_args()
    if args.config == "":
        parser.print_help()
        return
    args,seed,classes,last_epoch = read_config(args.config)

    n_epoch = 500
    n_units = args.units
    batch_size = args.batchsize
    gpu = args.gpu
    models_base_folder = "saved_models"
    output_folder = os.path.join("results",args.folder)  # args.folder  #R"C:\Users\bms\PycharmProjects\stylemotery_code" #
    exper_name = args.name
    dataset_folder = os.path.join("dataset",args.dataset)
    seperate_trees = args.seperate
    model_name = args.model
    layers = args.layers
    dropout = args.dropout
    cell = args.cell
    residual = args.residual

    output_file = open(os.path.join(output_folder, exper_name + "_results.txt"), mode="a")

    trees, tree_labels, lable_problems, tree_nodes = parse_src_files(dataset_folder,seperate_trees=seperate_trees)
    if model_name == "lstm":
        model = RecursiveLSTM(n_units, len(classes), layers=layers, dropout=dropout,feature_dict=tree_nodes, classes=classes, cell=cell,residual=residual)
    elif model_name == "bilstm":
        model = RecursiveBiLSTM(n_units, len(classes), layers=layers, dropout=dropout,feature_dict=tree_nodes, classes=classes,cell=cell,residual=residual)
    elif model_name == "treelstm":
        model = RecursiveTreeLSTM(n_children=layers, n_units=n_units,n_label=len(classes), dropout=dropout,feature_dict=tree_nodes, classes=classes)
    else:
        print("No model was found")
        return

    # load the model
    model_saved_name = "{0}_epoch_".format(exper_name)
    saved_models = [m for m in os.listdir(models_base_folder) if m.startswith(model_saved_name) and m.endswith(".my")]
    if len(saved_models) > 0:
        #pick the best one
        model_saved_name = list(sorted(saved_models,key=lambda name: int(name.split(".")[0].split("_")[-1]),reverse=True))[0]
    else:
        print("No model was found to load")
        return
    path = os.path.join(models_base_folder, model_saved_name)
    serializers.load_npz(path, model)

    if gpu >= 0:
        model.to_gpu()

    # Setup optimizer
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)#Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)#AdaGrad(lr=0.01)#NesterovAG(lr=0.01, momentum=0.9)#AdaGrad(lr=0.01) # MomentumSGD(lr=0.01, momentum=0.9)  # AdaGrad(lr=0.1) #
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))
    optimizer.add_hook(chainer.optimizer.GradientClipping(10.0))

    # load the optimizor
    opt_saved_name = "{0}_epoch_".format(exper_name)
    saved_opts = [m for m in os.listdir(models_base_folder) if m.startswith(opt_saved_name) and m.endswith(".opt")]
    if len(saved_opts) > 0:
        #pick the best one
        opt_saved_name = list(sorted(saved_opts,key=lambda name: int(name.split(".")[0].split("_")[-1]),reverse=True))[0]
    else:
        print("No model was found to load")
        return
    path = os.path.join(models_base_folder, opt_saved_name)
    serializers.load_npz(path, optimizer)

    trees, tree_labels = pick_subsets(trees, tree_labels, classes=classes)
    train_trees, train_lables, test_trees, test_lables, classes, cv = split_trees(trees, tree_labels, n_folds=5,
                                                                                  shuffle=True, seed=seed,
                                                                                  iterations=args.iterations)

    best_scores = (-1, -1, -1)  # (epoch, loss, accuracy)

    for epoch in range(last_epoch, n_epoch + 1):
        print('Epoch: {0:d} / {1:d}'.format(epoch, n_epoch))
        print("optimizer lr = ", optimizer.lr)
        print('Train')
        training_accuracy, training_loss = train(model, train_trees, train_lables, optimizer, batch_size, shuffle=True)
        print('Test')
        test_accuracy, test_loss = evaluate(model, test_trees, test_lables, batch_size)
        print()

        # save the best models
        saved = False
        if args.save > 0 and epoch > 0:
            epoch_, loss_, acc_ = best_scores
            # save the model with best accuracy or same accuracy and less loss
            if test_accuracy > acc_ or (test_accuracy >= acc_ and test_loss <= loss_):
                # remove last saved model
                remove_old_model(models_base_folder,exper_name, epoch_)
                # save models
                save_new_model(model,optimizer,models_base_folder,exper_name, epoch)
                saved = True
                print("saving ... ")
                best_scores = (epoch, test_loss, test_accuracy)

        output_file.write(
            "{0:<10}{1:<20.10f}{2:<20.10f}{3:<20.10f}{4:<20.10f}{5:<20.10f}{6:<10}\n".format(epoch,optimizer.lr, training_loss, test_loss,
                                                                                  training_accuracy, test_accuracy,
                                                                                  "saved" if saved else ""))
        output_file.flush()

        if epoch >= 5 and (test_loss < 0.001 or test_accuracy >= 1.0):
            output_file.write("\tEarly Stopping\n")
            print("\tEarly Stopping")
            break

            # if epoch is 3:
            #     lr = optimizer.lr
            #     setattr(optimizer, 'lr', lr / 10)

    output_file.close()


if __name__ == "__main__":
    main_experiment()
