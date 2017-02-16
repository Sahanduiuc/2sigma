import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.metrics import r2_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# import resource
# resource.setrlimit(resource.RLIMIT_AS, (2**33 - 2**31, 2**33 - 2**31))
import gc


def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight,
                  multioutput=multioutput)
    r = (np.sign(r2) * np.sqrt(np.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r

import kagglegym
env = kagglegym.make()
observation = env.reset()
train = observation.train

# train = pd.read_hdf('../input/train.h5')

low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

excl = ['id', 'sample', 'timestamp']
col = [c for c in train.columns if c not in excl]
train = train.loc[y_is_within_cut, col]
y = train.y
train.drop('y', axis=1, inplace=True)
train = train.fillna(-100)

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

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)
train_data = lgb.Dataset(X_train, y_train)
gc.collect()
bst = lgb.train(params, train_data, feval=r_score, verbose_eval=verbose)

df_fi = pd.DataFrame(bst.feature_name(), columns=['feature'])
df_fi['importance'] = list(bst.feature_importance())
df_fi.sort_values('importance', ascending=False, inplace=True)
print(df_fi)
plt.figure()
df_fi.head(10).plot(kind='barh',
                    x='feature',
                    y='importance',
                    sort_columns=False,
                    legend=False,
                    figsize=(10, 6),
                    facecolor='#1DE9B6',
                    edgecolor='white')

plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.show()
# feature  importance
# 93     technical_30         147
# 82     technical_17         133
# 85     technical_20         115
# 103    technical_40         100
# 86     technical_21          93
# 70      technical_2          80
# 96     technical_33          72
# 68      technical_0          69
# 90     technical_27          60
# 77     technical_11          53
# 104    technical_41          50
# 94     technical_31          50
# 71      technical_3          49
# 84     technical_19          49
# 66   fundamental_62          48
# 72      technical_5          48
# 25   fundamental_21          47
# 98     technical_35          46
# 107    technical_44          45
# 89     technical_25          45
# 88     technical_24          42
# 57   fundamental_53          42
# 99     technical_36          40
# 73      technical_6          40
# 79     technical_13          37
# 48   fundamental_44          33
# 69      technical_1          32
# 11    fundamental_7          32
# 52   fundamental_48          31
# 37   fundamental_33          30
# ..              ...         ...
# 28   fundamental_24          14
# 16   fundamental_12          14
# 47   fundamental_43          13
# 60   fundamental_56          13
# 3         derived_3          13
# 8     fundamental_3          13
# 64   fundamental_60          12
# 31   fundamental_27          11
# 81     technical_16          10
# 7     fundamental_2          10
# 21   fundamental_17           9
# 32   fundamental_28           9
# 17   fundamental_13           8
# 35   fundamental_31           8
# 29   fundamental_25           8
# 102    technical_39           7
# 75      technical_9           7
# 20   fundamental_16           6
# 87     technical_22           5
# 83     technical_18           5
# 67   fundamental_63           5
# 44   fundamental_40           5
# 42   fundamental_38           4
# 10    fundamental_6           2
# 97     technical_34           2
# 76     technical_10           2
# 30   fundamental_26           2
# 65   fundamental_61           2
# 61   fundamental_57           2
# 6     fundamental_1           1


if verbose:
    y_pred = bst.predict(X_test)
    print('\nScore for another fold: ', r_score(y_test, y_pred))
#
# train_data = lgb.Dataset(train, y)
# gc.collect()
# bst = lgb.train(params, train_data, feval=r_score, verbose_eval=verbose)


# while True:
#     test = observation.features[col]
#     # n = test.isnull().sum(axis=1)
#     # for c in test.columns:
#         # test[c + '_nan_'] = pd.isnull(test[c])
#     test = test.fillna(-100)
#     # test['znull'] = n
#     pred = observation.target
#     pred['y'] = bst.predict(test)
#
#     observation, reward, done, info = env.step(pred)
#     if done:
#         print("final ...", info["public_score"])
#         break
#     if observation.features.timestamp[0] % 10 == 0:
#         print(reward)
