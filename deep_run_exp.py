import argparse
import collections
import os
import random
from operator import itemgetter

import chainer
from chainer import optimizers
import math
# from deep_ast.tree_lstm.treelstm import TreeLSTM
from chainer import serializers
from models.lstm_models import RecursiveLSTM, RecursiveBiLSTM, RecursiveResidualLSTM
from models.clstm_models import RecursiveDyanmicLSTM
from models.tree_models import RecursiveTreeLSTM
from utils.exp_utlis import pick_subsets, split_trees,train,evaluate, read_config
from utils.dataset_utils import parse_src_files, print_model, unified_ast_trees, make_binary_tree


def print_table(table):
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    for line in table:
        print("| " + " | ".join("{:{}}".format(x, col_width[i])
                                for i, x in enumerate(line)) + " |")


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
    parser.add_argument('--train', '-t', type=str, default="", help='Experiment Training data info')

    parser.add_argument('--name', '-n', type=str, default="default_experiment", help='Experiment name')
    parser.add_argument('--dataset', '-d', type=str, default="cpp", help='Experiment dataset')
    parser.add_argument('--classes', '-c', type=int, default=-1, help='How many classes to include in this experiment')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--folder', '-f', type=str, default="", help='Base folder for logs and results')
    parser.add_argument('--batchsize', '-b', type=int, default=1, help='Number of examples in each mini batch')
    parser.add_argument('--layers', '-l', type=int, default=1, help='Number of Layers for LSTMs')
    parser.add_argument('--dropout', '-dr', type=float, default=0.2, help='Number of Layers for LSTMs')
    parser.add_argument('--iterations', '-i', type=int, default=0, help='CV iterations')

    parser.add_argument('--model', '-m', type=str, default="bilstm", help='Model used for this experiment')
    parser.add_argument('--units', '-u', type=int, default=100, help='Number of hidden units')
    parser.add_argument('--save', '-s', type=int, default=1, help='Save best models')

    args = parser.parse_args()

    n_epoch = 500
    n_units = args.units
    batch_size = args.batchsize
    gpu = args.gpu
    models_base_folder = "saved_models"
    output_folder = os.path.join("results",args.folder)  # args.folder  #R"C:\Users\bms\PycharmProjects\stylemotery_code" #
    exper_name = args.name
    dataset_folder = os.path.join("dataset",args.dataset)
    model_name = args.model
    layers = args.layers
    dropout = args.dropout

    trees, tree_labels, lable_problems, tree_nodes = parse_src_files(dataset_folder)

    if args.train:
        rand_seed, classes = read_config(os.path.join("train",args.dataset,args.train))
        trees, tree_labels = pick_subsets(trees, tree_labels, classes=classes)
    else:
        rand_seed = random.randint(0, 4294967295)
        if args.classes > -1:
            trees, tree_labels = pick_subsets(trees, tree_labels, labels=args.classes,seed=rand_seed,classes=None)

    #if model_name in ("treelstm","slstm"):
    # trees = make_binary_tree(unified_ast_trees(trees),2)

    train_trees, train_lables, test_trees, test_lables, classes, cv = split_trees(trees, tree_labels, n_folds=5,shuffle=True,seed=rand_seed,
                                                                                  iterations=args.iterations)

    output_file = open(os.path.join(output_folder, exper_name + "_results.txt"), mode="+w")
    output_file.write("Testing the model on all the datasets\n")
    output_file.write("Args :- " + str(args) + "\n")
    output_file.write("Seed :- " + str(rand_seed) + "\n")

    output_file.write("Classes :- (%s)\n" % [(idx, c) for idx, c in enumerate(classes)])
    output_file.write("Class ratio :- %s\n" % list(
        sorted([(t, c, c / len(tree_labels)) for t, c in collections.Counter(tree_labels).items()], key=itemgetter(0),
               reverse=False)))
    output_file.write("Cross Validation :-%s\n" % cv)
    output_file.write("Train labels :- (%s,%s%%): %s\n" % (
    len(train_lables), (len(train_lables) / len(tree_labels)) * 100, train_lables))
    output_file.write(
        "Test  labels :- (%s,%s%%): %s\n" % (len(test_lables), (len(test_lables) / len(tree_labels)) * 100, test_lables))
    if model_name == "lstm":
        model = RecursiveLSTM(n_units, len(classes), layers=layers, dropout=dropout,feature_dict=tree_nodes, classes=classes, peephole=False)
    elif model_name == "dlstm":
        model = RecursiveDyanmicLSTM(n_units, len(classes), layers=layers, dropout=dropout,feature_dict=tree_nodes, classes=classes, peephole=False)
    elif model_name == "bilstm":
<<<<<<< HEAD
        model = RecursiveBiLSTM(n_units, len(classes), layers=layers, dropout=dropout, classes=classes,peephole=False)
    elif model_name == "bbilstm":
        model = RecursiveBBiLSTM(n_units, len(classes), dropout=dropout, classes=classes)
=======
        model = RecursiveBiLSTM(n_units, len(classes), layers=layers, dropout=dropout,feature_dict=tree_nodes, classes=classes,peephole=False)
>>>>>>> 756f17db619b01a162b995803d305bb1dff7ef97
    elif model_name == "biplstm":
        model = RecursiveBiLSTM(n_units, len(classes), dropout=dropout, classes=classes,feature_dict=tree_nodes,peephole=True)
    elif model_name == "plstm":
        model = RecursiveLSTM(n_units, len(classes), layers=layers, dropout=dropout, classes=classes,feature_dict=tree_nodes, peephole=True)
    # elif model_name == "highway":
    #     model = RecursiveHighWayLSTM(n_units, len(classes),layers=layers, dropout=dropout, classes=classes, peephole=False)
    elif model_name == "reslstm":
        model = RecursiveResidualLSTM(n_units, len(classes),layers=layers, dropout=dropout, classes=classes,feature_dict=tree_nodes, peephole=False)
    elif model_name == "treelstm":
        model = RecursiveTreeLSTM(n_children=layers, n_units=n_units,n_label=len(classes), dropout=dropout,feature_dict=tree_nodes, classes=classes)
    else:
        print("No model was found")
        return

    output_file.write("Model:  {0}\n".format(exper_name))
    output_file.write("Params: {:,} \n".format(model.params_count()))
    output_file.write("        {0} \n".format(type(model).__name__))
    print_model(model, depth=1, output=output_file)

    if gpu >= 0:
        model.to_gpu()

    # Setup optimizer
    optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)#)daGrad(lr=0.01)
    #MomentumSGD(lr=0.01, momentum=0.9)#Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)#AdaGrad(lr=0.01)#NesterovAG(lr=0.01, momentum=0.9)#AdaGrad(lr=0.01) # MomentumSGD(lr=0.01, momentum=0.9)  # AdaGrad(lr=0.1) #
    output_file.write("Optimizer: {0} ".format((type(optimizer).__name__, optimizer.__dict__)))
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))

    hooks = [(k, v.__dict__) for k, v in optimizer._hooks.items()]
    output_file.write(" {0} \n".format(hooks))

    output_file.write("Evaluation\n")
    output_file.write("{0:<10}{1:<20}{2:<20}{3:<20}{4:<20}{5:<20}\n".format("epoch","learing_rate", "train_loss", "test_loss","train_accuracy", "test_accuracy"))
    output_file.flush()


    def drop_decay(epoch,initial_lrate=0.1,drop=0.5,epochs_drop = 10.0):
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    def time_decay(epoch, initial_lrate=0.1, decay=0.1):
        return initial_lrate * 1 / (1 + decay * epoch)

    def range_decay(epoch):
        class RangeDictionary(dict):
            def __getitem__(self, key):
                for r in self.keys():
                    if key in r:
                        return super().__getitem__(r)
                return super().__getitem__(key)
        rates = {range(0,10):0.01,
                 range(10,30):0.005,
                 range(30,50):0.001,
                 range(50,100):0.0005,
                 range(100,500):0.0001}
        return RangeDictionary(rates)[epoch]

    best_scores = (-1, -1, -1)  # (epoch, loss, accuracy)
    for epoch in range(1, n_epoch + 1):
        # optimizer.lr = range_decay(epoch-1)
        print('Epoch: {0:d} / {1:d}'.format(epoch, n_epoch))
        print("optimizer lr = ", optimizer.lr)
        print('Train')
        training_accuracy, training_loss = train(model, train_trees, train_lables, optimizer, batch_size, shuffle=True)
        print('Test')
        test_accuracy, test_loss = evaluate(model, test_trees, test_lables, batch_size)
        print()

        # if epoch % 5 == 0:
        #     model.curr_layers += 1
        # save the best models
        saved = False
        if args.save > 0 and epoch > 0:
            epoch_, loss_, acc_ = best_scores
            # save the model with best accuracy or same accuracy and less loss
            if test_accuracy > acc_ or (test_accuracy >= acc_ and test_loss <= loss_):
                # remove last saved model
                remove_old_model(models_base_folder,exper_name, epoch)
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

