import os

import chainer
from chainer import optimizers
# from deep_ast.tree_lstm.treelstm import TreeLSTM
from chainer import serializers
from models.lstm_models import RecursiveHighWayLSTM, RecursiveLSTM, RecursiveBiLSTM, RecursiveResidualLSTM
from models.tree_models import RecursiveTreeLSTM
from utils.exp_utlis import split_trees, pick_subsets, evaluate, train
from utils.fun_utils import parse_src_files, print_model
import argparse
from argparse import Namespace

def print_table(table):
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print("| " + " | ".join("{:{}}".format(x, col_width[i])
                                for i, x in enumerate(line)) + " |")

def read_config(filename):
    with open(filename) as file:
        for line in file:
            if line.startswith("Args "):
                args_line = line.split(":-",1)
                args = eval(args_line[1])
            elif line.startswith("Seed "):
                args_line = line.split(":-",1)
                seed = int(args_line[1].strip())
            elif line.startswith("Classes "):
                classes = [v for idx, v in eval(line.split(":-")[1])]
            # elif line.startswith("Train labels "):
            #     train_lables = [v for idx, v in eval(line.split(":")[1])]
            # elif line.startswith("Test labels "):
            #     test_trees = [v for idx, v in eval(line.split(":")[1])]
        return args,seed,classes

def main_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="", help='Configuration file')

    args = parser.parse_args()
    if args.config == "":
        parser.print_help()
        return
    args,seed,classes = read_config(args.config)

    n_epoch = 500
    n_units = args.units
    batch_size = args.batchsize
    gpu = args.gpu
    models_base_folder = "saved_models"
    output_folder = os.path.join("results",args.folder)  # args.folder  #R"C:\Users\bms\PycharmProjects\stylemotery_code" #
    exper_name = args.name
    dataset_folder = args.dataset
    model_name = args.model
    layers = args.layers
    dropout = args.dropout
    iterations = args.iterations

    output_file = open(os.path.join(output_folder, exper_name + "_results.txt"), mode="a")

    trees, tree_labels, lable_problems = parse_src_files(dataset_folder)
    if args.classes > -1:
        trees, tree_labels = pick_subsets(trees, tree_labels, labels=args.classes,classes=classes)
    train_trees, train_lables, test_trees, test_lables, classes, cv = split_trees(trees, tree_labels, n_folds=5,
                                                                                  shuffle=True,iterations=iterations)
    if model_name == "lstm":
        model = RecursiveLSTM(n_units, len(classes), layers=layers, dropout=dropout, classes=classes, peephole=False)
    elif model_name == "bilstm":
        model = RecursiveBiLSTM(n_units, len(classes), dropout=dropout, classes=classes,peephole=False)
    elif model_name == "biplstm":
        model = RecursiveBiLSTM(n_units, len(classes), dropout=dropout, classes=classes,peephole=True)
    elif model_name == "plstm":
        model = RecursiveLSTM(n_units, len(classes), layers=layers, dropout=dropout, classes=classes, peephole=True)
    elif model_name == "highway":
        model = RecursiveHighWayLSTM(n_units, len(classes),layers=layers, dropout=dropout, classes=classes, peephole=False)
    elif model_name == "reslstm":
        model = RecursiveResidualLSTM(n_units, len(classes),layers=layers, dropout=dropout, classes=classes, peephole=False)
    elif model_name == "treestm":
        model = RecursiveTreeLSTM(2, n_units, len(classes), classes=classes)
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

    best_scores = (-1, -1, -1)  # (epoch, loss, accuracy)
    for epoch in range(1, n_epoch + 1):
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
                model_saved_name = "{0}_epoch_{1}.my".format(exper_name, epoch_)
                path = os.path.join(models_base_folder, model_saved_name)
                if os.path.exists(path):
                    os.remove(path)
                # save models
                model_saved_name = "{0}_epoch_{1}.my".format(exper_name, epoch)
                path = os.path.join(models_base_folder, model_saved_name)
                serializers.save_npz(path, model)
                # save optimizer
                model_saved_name = "{0}_epoch_{1}.opt".format(exper_name, epoch)
                path = os.path.join(models_base_folder, model_saved_name)
                serializers.save_npz(path, optimizer)
                saved = True
                print("saving ... ")
                best_scores = (epoch, test_loss, test_accuracy)

        output_file.write(
            "{0:<10}{1:<20.10f}{2:<20.10f}{3:<20.10f}{4:<20.10f}{5:<10}\n".format(epoch, training_loss, test_loss,
                                                                                  training_accuracy, test_accuracy,
                                                                                  "saved" if saved else ""))
        output_file.flush()

        if epoch > 10 and (test_loss < 0.001 or test_accuracy >= 1.0):
            output_file.write("\tEarly Stopping\n")
            print("\tEarly Stopping")
            break

            # if epoch is 3:
            #     lr = optimizer.lr
            #     setattr(optimizer, 'lr', lr / 10)

    output_file.close()


if __name__ == "__main__":
    main_experiment()
