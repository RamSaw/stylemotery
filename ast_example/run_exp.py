import os
from collections import defaultdict, Counter
from pprint import pprint
from time import time

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from ast_example.ASTVectorizater import ASTVectorizer
from ast_example.InformationGain import TopRandomTreesEmbedding
from sys import platform as _platform


def get_basefolder():
    if _platform == "linux" or _platform == "linux2":
        return R"/home/bms/projects/stylometory/stylemotery/dataset700"
    elif _platform == "win32":
        return R"C:\Users\bms\PycharmProjects\stylemotery_code\dataset700"


def read_py_files(basefolder):
    files = os.listdir(basefolder)
    files_noext = ['.'.join(s.split('.')[:-1]) for s in files]

    problems = [p.split('.')[0] for p in files_noext]
    users = [' '.join(p.split('.')[1:]) for p in files_noext]

    return np.array([os.path.join(basefolder, file) for file in files]), np.array(users), np.array(problems)


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


def main_relax(pipline, relax=15):
    basefolder = get_basefolder()
    X, y, tags = read_py_files(basefolder)

    print("\t\t%s problems, %s users :" % (len(set(tags)), len(set(y))))

    folds = StratifiedKFold(y, n_folds=10)
    accuracy = []
    for idx, (train, test) in enumerate(folds):
        pipline.fit(X[train], y[train])
        y_predict_prob = pipline.predict_proba(X[test])
        classes_ = pipline.steps[-1][1].classes_
        y_predict_indices = y_predict_prob.argsort(axis=1)[:, ::-1][:, :relax]
        y_predict = []
        for i, predict in enumerate(y_predict_indices):
            y_predict_all = classes_[predict]
            target = y[test][i]
            if (target == y_predict_all).any():
                y_predict.append(target)
            else:
                y_predict.append(y_predict_all[0])

        y_predict = np.array(y_predict)
        accuracy.append(accuracy_score(y[test], y_predict))
        print("\t\t\taccuracy = ", accuracy[-1])
        # for feature in np.nonzero(pipline.steps[-1][1].feature_importances_)[0]:
        #     import_features[feature] += 1

    print("\tAVG =", np.mean(accuracy))
    # print("Features =",Counter(import_features).most_common(100))


def main(pipline):
    basefolder = get_basefolder()
    X, y, tags = read_py_files(basefolder)

    print("%s problems, %s users :" % (len(set(tags)), len(set(y))))

    folds = StratifiedKFold(y, n_folds=5)
    accuracy = []
    import_features = defaultdict(int)
    features = []
    for idx, (train, test) in enumerate(folds):
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
        print()

    print("features categories =", [(k, v / float(len(features)) * 100.0) for k, v in Counter(features).most_common()])
    print("AVG =", np.mean(accuracy))
    # print("Features =",Counter(import_features).most_common(100))


def main_gridsearch():
    basefolder = get_basefolder()
    X, y, tags = read_py_files(basefolder)

    print("%s problems, %s users :" % (len(set(tags)), len(set(y))))
    pipline = Pipeline([
        ('ast', ASTVectorizer(dtype=np.float32)),
        ('select', TopRandomTreesEmbedding()),  # PredefinedFeatureSelection()),
        ('clf', RandomForestClassifier())])

    folds = StratifiedKFold(y, n_folds=5)
    parameters = {
        'ast__ngrams': (2, 3),
        'ast__normalize': (True, False),
        'ast__idf': (True, False),

        'select__k': (500, 700, 1000),
        'select__n_estimators': (500, 1000, 2000),
        'select__max_depth': (20, 40, 60),

        'clf__n_estimators': (100, 500, 800, 1000),
        'clf__max_features': ('log2', 'sqrt'),
        'clf__criterion': ('gini', 'entropy'),
    }

    grid_search = GridSearchCV(estimator=pipline, param_grid=parameters, cv=folds, n_jobs=2)
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


if __name__ == "__main__":
    # main_gridsearch()
    relax_list = [1, 5, 10, 15]
    k_list = [700, 900, 1000]
    for i in relax_list:
        print("Relax = ", i)
        for k in k_list:
            print("\tk = ", k)
            pipline = Pipeline([
                ('astvector', ASTVectorizer(ngram=3, normalize=True, idf=True, dtype=np.float32)),
                ('selection', TopRandomTreesEmbedding(k=k, n_estimators=1000, max_depth=40)),
                # PredefinedFeatureSelection()),
                ('randforest', RandomForestClassifier(n_estimators=500, max_features="auto"))])
            main_relax(pipline, relax=i)

            # print("relax")
            # pipline = Pipeline([
            #     ('astvector', ASTVectorizer(ngram=2, normalize=True, idf=True, dtype=np.float32)),
            #     ('selection', TopRandomTreesEmbedding(k=700, n_estimators=1000, max_depth=40)),
            #     # PredefinedFeatureSelection()),
            #     ('randforest', RandomForestClassifier(n_estimators=500, max_features="auto"))])
            # main_relax(pipline, relax=1)
            # print("predict")
            # main(pipline)
