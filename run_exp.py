import argparse
import random
import traceback
from collections import defaultdict, Counter
from operator import itemgetter
from pprint import pprint
from time import time
import os
import numpy as np
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from ast_tree.tree_nodes import DotNodes
from information_gain.InformationGain import TopRandomTreesEmbedding
from ast_tree.ASTVectorizater import ASTVectorizer
from utils.exp_utlis import read_train_config, pick_subsets, split_trees
from utils.dataset_utils import parse_src_files
import collections

def full_evaluation(rf, X, y, cv):
    precision = []
    accuracy = []
    sensitivity = []
    matthews = []
    r2 = []
    f1 = []
    auroc = []
    cm = [[0, 0], [0, 0]]

    for i, (train, test) in enumerate(cv):
        probas_ = rf.fit(X[train], y[train]).predict_proba(X[test])
        classes = rf.fit(X[train], y[train]).predict(X[test])
        r2 = np.append(r2, (r2_score(y[test], probas_[:, 1])))
        precision = np.append(precision, (precision_score(y[test], classes)))
        auroc = np.append(auroc, (roc_auc_score(y[test], classes)))
        accuracy = np.append(accuracy, (accuracy_score(y[test], classes)))
        sensitivity = np.append(sensitivity, (recall_score(y[test], classes)))
        f1 = np.append(f1, (f1_score(y[test], classes)))
        matthews = np.append(matthews, (matthews_corrcoef(y[test], classes)))
        cma = np.add(cma, (confusion_matrix(y[test], classes)))

    cma = np.array(cma)
    r2 = np.array(r2)
    precision = np.array(precision)
    accuracy = np.array(accuracy)
    sensitivity = np.array(sensitivity)
    f1 = np.array(f1)
    auroc = np.array(auroc)
    matthews = np.array(matthews)

    print("KF Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))
    print("KF Precision: %0.2f (+/- %0.2f)" % (precision.mean(), precision.std() * 2))
    print("KF Sensitivity: %0.2f (+/- %0.2f)" % (sensitivity.mean(), sensitivity.std() * 2))
    print("KF R^2: %0.2f (+/- %0.2f)" % (r2.mean(), r2.std() * 2))
    print("KF F1: %0.2f (+/- %0.2f)" % (f1.mean(), f1.std() * 2))
    print("KF AUROC: %0.2f (+/- %0.2f)" % (auroc.mean(), auroc.std() * 2))
    print("KF Matthews: %0.2f (+/- %0.2f)" % (matthews.mean(), matthews.std() * 2))
    print("Confusion Matrix", cma)


def exp_relax(pipline,X,y,tags, relax=[],cv=None,output=sys.stdout):
    # basefolder = get_basefolder()
    # X, y, tags = parse_src_files(basefolder)

    # X,y,tags = split_trees2(X, y, tags)

    print("\t\t%s unique problems, %s unique users : " % (len(set(tags)), len(set(y))))
    print("\t\t%s all problems, %s all users : " % (len(tags), len(y)))
    # ratio = [(i, Counter(y)[i] / float(len(y)) * 100.0) for i in Counter(y).most_common()]
    # print("\t\t all users ratio ",ratio)

    accuracy = [[] for i in relax]
    for idx, (train, test) in enumerate(cv.split(X,y)):
        pipline.fit(X[train], y[train])
        y_predict_prob = pipline.predict_proba(X[test])
        classes_ = pipline.steps[-1][1].classes_
        y_predict = [[] for i in relax]
        print("\t\t{0}/{1}".format(idx+1,cv.n_splits))
        for relax_i in relax:
            y_predict_indices = y_predict_prob.argsort(axis=1)[:, ::-1][:, :relax_i]
            for i, predict in enumerate(y_predict_indices):
                y_predict_all = classes_[predict]
                target = y[test][i]
                if (target == y_predict_all).any():
                    y_predict[relax_i-1].append(target)
                else:
                    y_predict[relax_i-1].append(y_predict_all[0])

        y_predict = np.array(y_predict)
        for i in relax:
            accuracy[i-1].append(accuracy_score(y[test], y_predict[i-1]))
            output.write("\t\t\t\trelax = %s accuracy = %s \n" % (i,accuracy[i-1][-1]))
            print("\t\t\t\trelax = %s accuracy = %s " % (i,accuracy[i-1][-1]))
        output.flush()
        # for feature in np.nonzero(pipline.steps[-1][1].feature_importances_)[0]:
        #     import_features[feature] += 1

    for i in relax:
        output.write("Relax = %s \tAccuracy = %s \n" % (i,np.mean(accuracy[i-1])))
        print("Relax = %s \tAccuracy = %s " % (i, np.mean(accuracy[i-1])))
    # print("Features =",Counter(import_features).most_common(100))


def main(pipline):
    basefolder = "dataset\cpp"
    X, y, tags,features = parse_src_files(basefolder)

    print("%s problems, %s users :" % (len(set(tags)), len(set(y))))

    folds = StratifiedKFold(n_splits=10)
    accuracy = []
    import_features = defaultdict(int)
    features = []
    for idx, (train, test) in enumerate(folds.split(X,y)):
        pipline.fit(X[train], y[train])
        y_predict = pipline.predict(X[test])
        accuracy.append(accuracy_score(y[test], y_predict))

        extract = pipline.steps[0][1]
        select = pipline.steps[1][1]
        rf = pipline.steps[2][1]

        print("accuracy = ", accuracy[-1])
        non_zero_features = np.nonzero(rf)[0]
        print("zero features =", len(rf.feature_importances_) - len(non_zero_features))
        print("Non zero features =", len(non_zero_features))
        for feature in np.nonzero(rf.feature_importances_)[0]:
            import_features[feature] += 1
        for f in select.important_indices:
            features.append(extract.features_categories[f])

    print("features categories =", [(k, v / float(len(features)) * 100.0) for k, v in Counter(features).most_common()])
    print("AVG =", np.mean(accuracy))
    # print("Features =",Counter(import_features).most_common(100))


def report(grid_scores, n_top=10):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print()
        print()
        print("Top {0} Models:".format(n_top))
        print("\tModel with rank: {0}".format(i + 1))
        print("\tMean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("\tParameters: {0}".format(score.parameters))
        print("")


def main_gridsearch():
    basefolder = get_basefolder()
    X, y, tags = parse_src_files(basefolder)

    print("%s problems, %s users :" % (len(set(tags)), len(set(y))))
    pipline = Pipeline([
        ('ast', ASTVectorizer(DotNodes(),normalize=True, idf=True, dtype=np.float32)),
        ('select', TopRandomTreesEmbedding()),  # PredefinedFeatureSelection()),
        ('clf', RandomForestClassifier())])

    folds = StratifiedKFold(y, n_folds=5)
    parameters = {
        'ast__ngram': (2,),
        'ast__v_skip': (1, 2),

        'select__k': (1000, 1200, 1500, 2000),
        'select__n_estimators': (1000, 1500),
        'select__max_depth': (40, 60),

        'clf__n_estimators': (1000, 1500),
        'clf__min_samples_split': (1,),
    }

    grid_search = GridSearchCV(estimator=pipline, param_grid=parameters, cv=folds, n_jobs=5, verbose=10)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    report(grid_search.grid_scores_)

def main_relax(X,y):
    relax_list = list(range(1,21))#[1, 5, 10, 15]
    k_list = [700, 900, 1000]
    for i in relax_list:
        print("Relax = ", i)
        for k in k_list:
            print("\tk = ", k)
            pipline = Pipeline([
                ('astvector', ASTVectorizer(ngram=3, v_skip=1, normalize=True, idf=True, dtype=np.float32)),
                ('selection', TopRandomTreesEmbedding(k=k, n_estimators=1000, max_depth=40)),
                # PredefinedFeatureSelection()),
                ('randforest', RandomForestClassifier(n_estimators=1000, min_samples_split=1, max_features="auto"))])
            exp_relax(pipline, relax=i)

def test_all():
    basefolder = get_basefolder()
    X, y, tags = parse_src_files(basefolder)
    try:
        ast_tree = ASTVectorizer(ngram=2, normalize=True, idf=True, norm="l2")
        ast_tree.fit(X, y)
    except:
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', type=str, default="70_authors.labels1.txt", help='Experiment Training data info')
    parser.add_argument('--name', '-n', type=str, default="random_forest_experiment_relax", help='Experiment name')
    parser.add_argument('--dataset', '-d', type=str, default="python", help='Experiment dataset')
    parser.add_argument('--classes', '-c', type=int, default=-1, help='How many classes to include in this experiment')
    parser.add_argument('--folds', '-fo', type=int, default=5, help='Number of folds')
    parser.add_argument('--folder', '-f', type=str, default="RF", help='Base folder for logs and results')

    # train_labels = [
    #                 ("RF_250_sep_05_labels1","5_authors.labels1.txt"),
    #                 ("RF_250_sep_05_labels2","5_authors.labels2.txt"),
    #                 ("RF_250_sep_05_labels3","5_authors.labels3.txt"),
    #                 ("RF_250_sep_05_labels4","5_authors.labels4.txt"),
    #                 ("RF_250_sep_05_labels5","5_authors.labels5.txt"),
    #                 ("RF_250_sep_25_labels1", "25_authors.labels1.txt"),
    #                 ("RF_250_sep_25_labels2", "25_authors.labels2.txt"),
    #                 ("RF_250_sep_25_labels3", "25_authors.labels3.txt"),
    #                 ("RF_250_sep_25_labels4", "25_authors.labels4.txt"),
    #                 ("RF_250_sep_25_labels5", "25_authors.labels5.txt")

    # train_labels = [
    #     ("RF_250_sep_10_labels1", "10_authors.labels1.txt"),
    #     ("RF_250_sep_10_labels2", "10_authors.labels2.txt"),
    #     ("RF_250_sep_10_labels3", "10_authors.labels3.txt"),
    #     ("RF_250_sep_10_labels4", "10_authors.labels4.txt"),
    #     ("RF_250_sep_10_labels5", "10_authors.labels5.txt"),
    #     ("RF_250_sep_15_labels1", "15_authors.labels1.txt"),
    #     ("RF_250_sep_15_labels2", "15_authors.labels2.txt"),
    #     ("RF_250_sep_15_labels3", "15_authors.labels3.txt"),
    #     ("RF_250_sep_15_labels4", "15_authors.labels4.txt"),
    #     ("RF_250_sep_15_labels5", "15_authors.labels5.txt")
    # ]

    # train_labels = [
    #     ("RF_500_70_labels1", "70_authors.labels1.txt"),
    # ]
    args = parser.parse_args()
    n_folds = args.folds
    exper_name = args.name
    output_folder = os.path.join("results",args.folder)  # args.folder  #R"C:\Users\bms\PycharmProjects\stylemotery_code" #
    dataset_folder = os.path.join("dataset", args.dataset)
    trees, tree_labels, lable_problems, features = parse_src_files(dataset_folder,seperate_trees=False)
    #print(len(trees))
    pipline = Pipeline([
        ('astvector', ASTVectorizer(features, ngram=2, v_skip=0, normalize=True, idf=True, dtype=np.float32)),
        ('selection', TopRandomTreesEmbedding(k=1000, n_estimators=1500, max_depth=20)),
        # PredefinedFeatureSelection()),
        ('randforest',RandomForestClassifier(n_estimators=1000, min_samples_split=2, max_features="auto", criterion="entropy"))])
    # ('randforest', xgboost.XGBClassifier(learning_rate=0.1,max_depth= 10,subsample=1.0, min_child_weight = 5,colsample_bytree = 0.2 ))])
    # exp_relax(pipline,trees,tree_labels,lable_problems, relax=1,cv=cv)

    # exper_name = model_name
    # args.train = train_file
    print()
    print(exper_name, flush=True)
    if args.train:
        rand_seed, classes = read_train_config(os.path.join("train", args.dataset.split("_")[0], args.train))
        trees_subset, tree_labels_subset = pick_subsets(trees, tree_labels, classes=classes)
        #print(tree_labels_subset)
        #print(classes)
        #print(rand_seed)
        #print(len(trees_subset))
    else:
        rand_seed = random.randint(0, 4294967295)
        if args.classes > -1:
            trees_subset, tree_labels_subset = pick_subsets(trees, tree_labels, labels=args.classes, seed=rand_seed,
                                                            classes=None)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rand_seed)
    output_file = open(os.path.join(output_folder, exper_name + "_results.txt"), mode="+w")
    output_file.write("Testing the model on all the datasets\n")
    output_file.write("Args :- " + str(args) + "\n")
    # output_file.write("Seed :- " + str(rand_seed) + "\n")
    output_file.write("Cross Validation :-%s\n" % cv)
    output_file.write("Classes :- (%s)\n" % [(idx, c) for idx, c in enumerate(set(tree_labels_subset))])
    output_file.write("Class ratio :- %s\n" % list(
        sorted([(t, c, c / len(tree_labels_subset)) for t, c in collections.Counter(tree_labels_subset).items()],
               key=itemgetter(0),
               reverse=False)))
    output_file.write("Model:  {0}\n".format(exper_name))
    pprint(pipline.steps, stream=output_file, indent=5, depth=2)
    #main_gridsearch()
    #main_relax()
    output_file.flush()
    exp_relax(pipline,trees_subset,tree_labels_subset,lable_problems,relax=list(range(1,20)),cv=cv,output=output_file)
    output_file.flush()
    print()
    output_file.close()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train', '-t', type=str, default="", help='Experiment Training data info')
    # parser.add_argument('--name', '-n', type=str, default="random_forest_experiment", help='Experiment name')
    # parser.add_argument('--dataset', '-d', type=str, default="python", help='Experiment dataset')
    # parser.add_argument('--classes', '-c', type=int, default=-1, help='How many classes to include in this experiment')
    # parser.add_argument('--folds', '-fo', type=int, default=5, help='Number of folds')
    # parser.add_argument('--folder', '-f', type=str, default="RF", help='Base folder for logs and results')
    #
    # # train_labels = [
    # #                 ("RF_250_sep_05_labels1","5_authors.labels1.txt"),
    # #                 ("RF_250_sep_05_labels2","5_authors.labels2.txt"),
    # #                 ("RF_250_sep_05_labels3","5_authors.labels3.txt"),
    # #                 ("RF_250_sep_05_labels4","5_authors.labels4.txt"),
    # #                 ("RF_250_sep_05_labels5","5_authors.labels5.txt"),
    # #                 ("RF_250_sep_25_labels1", "25_authors.labels1.txt"),
    # #                 ("RF_250_sep_25_labels2", "25_authors.labels2.txt"),
    # #                 ("RF_250_sep_25_labels3", "25_authors.labels3.txt"),
    # #                 ("RF_250_sep_25_labels4", "25_authors.labels4.txt"),
    # #                 ("RF_250_sep_25_labels5", "25_authors.labels5.txt")
    #
    # # train_labels = [
    # #     ("RF_250_sep_10_labels1", "10_authors.labels1.txt"),
    # #     ("RF_250_sep_10_labels2", "10_authors.labels2.txt"),
    # #     ("RF_250_sep_10_labels3", "10_authors.labels3.txt"),
    # #     ("RF_250_sep_10_labels4", "10_authors.labels4.txt"),
    # #     ("RF_250_sep_10_labels5", "10_authors.labels5.txt"),
    # #     ("RF_250_sep_15_labels1", "15_authors.labels1.txt"),
    # #     ("RF_250_sep_15_labels2", "15_authors.labels2.txt"),
    # #     ("RF_250_sep_15_labels3", "15_authors.labels3.txt"),
    # #     ("RF_250_sep_15_labels4", "15_authors.labels4.txt"),
    # #     ("RF_250_sep_15_labels5", "15_authors.labels5.txt")
    # # ]
    #
    # train_labels = [
    #     ("RF_500_70_labels1", "70_authors.labels1.txt"),
    # ]
    # args = parser.parse_args()
    # n_folds = args.folds
    # output_folder = os.path.join("results",args.folder)  # args.folder  #R"C:\Users\bms\PycharmProjects\stylemotery_code" #
    # dataset_folder = os.path.join("dataset", args.dataset)
    # trees, tree_labels, lable_problems, features = parse_src_files(dataset_folder,seperate_trees=False)
    # for model_name,train_file in train_labels:
    #     exper_name = model_name
    #     args.train = train_file
    #     print(exper_name,flush=True)
    #     if args.train:
    #         rand_seed, classes = read_train_config(os.path.join("train", args.dataset, args.train))
    #         trees_subset, tree_labels_subset = pick_subsets(trees, tree_labels, classes=classes)
    #     else:
    #         rand_seed = random.randint(0, 4294967295)
    #         if args.classes > -1:
    #             trees_subset, tree_labels_subset = pick_subsets(trees, tree_labels, labels=args.classes,seed=rand_seed,classes=None)
    #
    #     cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rand_seed)
    #     #main_gridsearch()
    #     #main_relax()
    #     pipline = Pipeline([
    #         ('astvector', ASTVectorizer(features,ngram=2,v_skip=0, normalize=True, idf=True, dtype=np.float32)),
    #         ('selection', TopRandomTreesEmbedding(k=700, n_estimators=900, max_depth=20)),
    #         # PredefinedFeatureSelection()),
    #         ('randforest', RandomForestClassifier(n_estimators=500,min_samples_split=2, max_features="auto",criterion="entropy"))])
    #         # ('randforest', xgboost.XGBClassifier(learning_rate=0.1,max_depth= 10,subsample=1.0, min_child_weight = 5,colsample_bytree = 0.2 ))])
    #     # exp_relax(pipline,trees,tree_labels,lable_problems, relax=1,cv=cv)
    #
    #     output_file = open(os.path.join(output_folder, exper_name + "_results.txt"), mode="+w")
    #     output_file.write("Testing the model on all the datasets\n")
    #     output_file.write("Args :- " + str(args) + "\n")
    #     output_file.write("Seed :- " + str(rand_seed) + "\n")
    #     output_file.write("Cross Validation :-%s\n" % cv)
    #     output_file.write("Classes :- (%s)\n" % [(idx, c) for idx, c in enumerate(set(tree_labels_subset))])
    #     output_file.write("Class ratio :- %s\n" % list(
    #         sorted([(t, c, c / len(tree_labels_subset)) for t, c in collections.Counter(tree_labels_subset).items()], key=itemgetter(0),
    #                reverse=False)))
    #     output_file.write("Model:  {0}\n".format(exper_name))
    #     pprint(pipline.steps,stream=output_file, indent=5,depth=2)
    #     output_file.flush()
    #     exp_relax(pipline,trees_subset,tree_labels_subset,lable_problems,relax=i,cv=cv,output=output_file)
    #     output_file.close()
    #     print()

    # # # print("predict")
    # main(pipline)
    # # test_all()
