from matplotlib import gridspec
from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from bayes_opt import BayesianOptimization

# Load data set and target values
from xgboost import XGBClassifier

from ast_example.ASTVectorizater import ASTVectorizer
from ast_example.InformationGain import TopRandomTreesEmbedding
from utils import get_basefolder, parse_src_files


# basefolder = get_basefolder()
# X, y, tags = parse_src_files(basefolder)


# def xgboostcv(max_depth,
#               learning_rate,
#               n_estimators,
#               gamma,
#               min_child_weight,
#               max_delta_step,
#               subsample,
#               colsample_bytree,
#               silent=True,
#               nthread=-1,
#               seed=1234):
#     return cross_val_score(XGBClassifier(max_depth=int(max_depth),
#                                          learning_rate=learning_rate,
#                                          n_estimators=int(n_estimators),
#                                          silent=silent,
#                                          nthread=nthread,
#                                          gamma=gamma,
#                                          min_child_weight=min_child_weight,
#                                          max_delta_step=max_delta_step,
#                                          subsample=subsample,
#                                          colsample_bytree=colsample_bytree,
#                                          seed=seed),
#                            X,y,
#                            "log_loss",
#                            cv=5).mean()
#
#
def rfccv(k, sn_estimators,n_estimators, min_samples_split, max_features):
    pipline = Pipeline([
        ('astvector', ASTVectorizer(ngram=2, normalize=True, idf=True, dtype=np.float32)),
        ('selection', TopRandomTreesEmbedding(k=k, n_estimators=sn_estimators, max_depth=40)),
        ('randforest', RFC(n_estimators=int(n_estimators),
                           min_samples_split=int(min_samples_split),
                           max_features=min(max_features, 0.999),
                           random_state=2))])
    return cross_val_score(pipline, data, target, 'f1', cv=5).mean()


if __name__ == "__main__":
    # svcBO = BayesianOptimization(xgboostcv, {'max_depth': (5, 10),
    #                                          'learning_rate': (0.01, 0.3),
    #                                          'n_estimators': (50, 1000),
    #                                          'gamma': (1., 0.01),
    #                                          'min_child_weight': (2, 10),
    #                                          'max_delta_step': (0, 0.1),
    #                                          'subsample': (0.7, 0.8),
    #                                          'colsample_bytree': (0.5, 0.99)
    #                                          })

    rfcBO = BayesianOptimization(rfccv, {'ngram=2': [2, 3],
                                         'k': (50, 1500),
                                         'sn_estimators' : (500,1500),
                                         'n_estimators': (10, 250),
                                         'min_samples_split': (2, 25),
                                         'max_features': (0.1, 0.999)})
    svcBO.maximize()

    print('-' * 53)
    rfcBO.maximize()

    print('-' * 53)
    print('Final Results')
    print('SVC: %f' % svcBO.res['max']['max_val'])
    print('RFC: %f' % rfcBO.res['max']['max_val'])
