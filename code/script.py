import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.metrics import r2_score
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
import resource
resource.setrlimit(resource.RLIMIT_AS, (2**33 - 2**31, 2**33 - 2**31))
resource.setrlimit(resource.RLIMIT_NPROC, (6, 6))
import gc

# import kagglegym
# env = kagglegym.make()
# observation = env.reset()
#
# excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
# col = [c for c in observation.train.columns if c not in excl]
# train = observation.train[col]
# y = observation.train.y

train = pd.read_hdf('../input/train.h5')
y = train.y
excl = ['id', 'sample', 'y', 'timestamp']
col = [c for c in train.columns if c not in excl]
train = train[col]

# d_mean = train.median(axis=0)
# n = train.isnull().sum(axis=1)
# for c in train.columns:
#     train[c + '_nan_'] = pd.isnull(train[c])
#     d_mean[c + '_nan_'] = 0
# train = train.fillna(d_mean)
# train['znull'] = n
# n = []

train = train.fillna(-100)


def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight,
                  multioutput=multioutput)
    r = (np.sign(r2) * np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r


verbose = 1
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': verbose
}

# kf = KFold(n_splits=3, shuffle=True)
# for train_index, test_index in kf.split(train.values):
#     X_train, X_test = train.values[train_index], train.values[test_index]
#     y_train, y_test = y.values[train_index], y.values[test_index]
#     train_data = lgb.Dataset(X_train, y_train)
#     bst = lgb.train(params, train_data, feval=r_score, verbose_eval=verbose)
#     if verbose:
#         y_pred = bst.predict(X_test)
#         print('\nScore for another fold: ', r_score(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, y_train)
gc.collect()
bst = lgb.train(params, train_data, feval=r_score, verbose_eval=verbose)
if verbose:
    y_pred = bst.predict(X_test)
    print('\nScore for another fold: ', r_score(y_test, y_pred))


# while True:
#     test = observation.features[col]
#     n = test.isnull().sum(axis=1)
#     for c in test.columns:
#         test[c + '_nan_'] = pd.isnull(test[c])
#     test = test.fillna(d_mean)
#     test['znull'] = n
#     pred = observation.target
#     pred['y'] = bst.predict(test)
#
#     observation, reward, done, info = env.step(pred)
#     if done:
#         print("el fin ...", info["public_score"])
#         break
#     if observation.features.timestamp[0] % 100 == 0:
#         print(reward)