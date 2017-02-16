import numpy as np
import pandas as pd
import random
from sklearn.metrics import r2_score
from sklearn import ensemble, linear_model, metrics
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
import gc
plt.style.use('ggplot')
pd.options.mode.chained_assignment = None


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
o = env.reset()
train = o.train

# train = pd.read_hdf('../input/train.h5')

low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
train = train.loc[y_is_within_cut, :]

d_mean = train.median(axis=0)
train["nbnulls"] = train.isnull().sum(axis=1)
col = [x for x in train.columns if x not in ['id', 'timestamp', 'y']]

rnd = 17

# keeping na information on some columns (best selected by the tree algorithms)
add_nas_ft = True
nas_cols = ['technical_9', 'technical_0', 'technical_32', 'technical_16', 'technical_38',
            'technical_44', 'technical_20', 'technical_30', 'technical_13']
# columns kept for evolution from one month to another (best selected by the tree algorithms)
add_diff_ft = True
diff_cols = ['technical_22', 'technical_20', 'technical_30', 'technical_13', 'technical_34']


# homemade class used to infer randomly on the way the model learns
class createLinearFeatures:
    def __init__(self, n_neighbours=1, max_elts=None, verbose=True, random_state=None):
        self.rnd = random_state
        self.n = n_neighbours
        self.max_elts = max_elts
        self.verbose = verbose
        self.neighbours = []
        self.clfs = []

    def fit(self, train, y):
        if self.rnd != None:
            random.seed(self.rnd)
        if self.max_elts == None:
            self.max_elts = len(train.columns)
        list_vars = list(train.columns)
        random.shuffle(list_vars)

        lastscores = np.zeros(self.n) + 1e15

        for elt in list_vars[:self.n]:
            self.neighbours.append([elt])
        list_vars = list_vars[self.n:]

        for elt in list_vars:
            indice = 0
            scores = []
            for elt2 in self.neighbours:
                if len(elt2) < self.max_elts:
                    clf = linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
                    clf.fit(train[elt2 + [elt]], y)
                    scores.append(metrics.mean_squared_error(y, clf.predict(train[elt2 + [elt]])))
                    indice = indice + 1
                else:
                    scores.append(lastscores[indice])
                    indice = indice + 1
            gains = lastscores - scores
            if gains.max() > 0:
                temp = gains.argmax()
                lastscores[temp] = scores[temp]
                self.neighbours[temp].append(elt)

        indice = 0
        for elt in self.neighbours:
            clf = linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
            clf.fit(train[elt], y)
            self.clfs.append(clf)
            if self.verbose:
                print(indice, lastscores[indice], elt)
            indice = indice + 1

    def transform(self, train):
        indice = 0
        for elt in self.neighbours:
            # this line generates a warning. Could be avoided by working and returning
            # with a copy of train.
            # kept this way for memory management
            train['neighbour' + str(indice)] = self.clfs[indice].predict(train[elt])
            indice = indice + 1
        return train

    def fit_transform(self, train, y):
        self.fit(train, y)
        return self.transform(train)


# a home-made class attempt to remove outliers by successive quantization on residuals
class recurrent_linear_approx():
    def __init__(self, quant=.999, limit_size_train=.9):
        self.quant = quant
        self.limit_size_train = limit_size_train
        self.bestmodel = []

    def fit(self, train, y):
        internal_model = linear_model.LinearRegression()
        bestscore = 1e15
        better = True
        indextrain = train.dropna().index
        limitlen = len(train) * self.limit_size_train
        while better:
            internal_model.fit(train.ix[indextrain], y.ix[indextrain])
            score = metrics.mean_squared_error(internal_model.predict(train.ix[indextrain]), y.ix[indextrain])
            if score < bestscore:
                bestscore = score
                self.bestmodel = internal_model
                residual = y.ix[indextrain] - internal_model.predict(train.ix[indextrain])
                indextrain = residual[abs(residual) <= abs(residual).quantile(self.quant)].index
                if len(indextrain) < limitlen:
                    better = False
            else:
                better = False
                self.bestmodel = internal_model

    def predict(self, test):
        return self.bestmodel.predict(test)


if add_nas_ft:
    for elt in nas_cols:
        train[elt + '_na'] = pd.isnull(train[elt]).apply(lambda x: 1 if x else 0)
        # no need to keep columns with no information
        if len(train[elt + '_na'].unique()) == 1:
            print("removed:", elt, '_na')
            del train[elt + '_na']
            nas_cols.remove(elt)

if add_diff_ft:
    train = train.sort_values(by=['id', 'timestamp'])
    for elt in diff_cols:
        # a quick way to obtain deltas from one month to another but it is false on the first
        # month of each id
        train[elt + "_d"] = train[elt].rolling(2).apply(lambda x: x[1] - x[0]).fillna(0)
    # removing month 0 to reduce the impact of erroneous deltas
    train = train[train.timestamp != 0]

print(train.shape)
cols = [x for x in train.columns if x not in ['id', 'timestamp', 'y', 'sample']]

# generation of linear models
cols2fit = ['technical_22', 'technical_20', 'technical_30_d', 'technical_20_d', 'technical_30',
            'technical_13', 'technical_34']
models = []
columns = []
residuals = []
for elt in cols2fit:
    print("fitting linear model on ", elt)
    model = recurrent_linear_approx(quant=.99, limit_size_train=.9)
    model.fit(train.loc[:, [elt]], train.loc[:, 'y'])
    models.append(model)
    columns.append([elt])
    residuals.append(abs(model.predict(train[[elt]].fillna(d_mean)) - train.y))

train = train.fillna(d_mean)

# adding all trees generated by a tree regressor
print("adding new features")
featureexpander = createLinearFeatures(n_neighbours=30, max_elts=2, verbose=True, random_state=rnd)
index2use = train[abs(train.y) < 0.086].index
featureexpander.fit(train.ix[index2use, cols], train.ix[index2use, 'y'])
trainer = featureexpander.transform(train[cols])
treecols = trainer.columns

# print("training trees")
# model = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=rnd, verbose=0)
# model.fit(trainer, train.y)
# print(pd.DataFrame(model.feature_importances_, index=treecols).sort_values(by=[0]).tail(30))
# for elt in model.estimators_:
#     models.append(elt)
#     columns.append(treecols)
#     residuals.append(abs(elt.predict(trainer) - train.y))
#
# # model selection : create a new target selecting models with lowest asolute residual for each line
# # the objective at this step is to keep only the few best elements which should
# # lead to a better generalization
# num_to_keep = 10
# targetselector = np.array(residuals).T
# targetselector = np.argmin(targetselector, axis=1)
# print("selecting best models:")
# print(pd.Series(targetselector).value_counts().head(num_to_keep))
#
# tokeep = pd.Series(targetselector).value_counts().head(num_to_keep).index
# tokeepmodels = []
# tokeepcolumns = []
# tokeepresiduals = []
# for elt in tokeep:
#     tokeepmodels.append(models[elt])
#     tokeepcolumns.append(columns[elt])
#     tokeepresiduals.append(residuals[elt])
#
# # creating a new target for a model in charge of predicting which model is best for the current line
# targetselector = np.array(tokeepresiduals).T
# targetselector = np.argmin(targetselector, axis=1)
#
# print("training selection model")
# modelselector = ensemble.ExtraTreesClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=rnd, verbose=0)
# modelselector.fit(trainer, targetselector)
# print(pd.DataFrame(modelselector.feature_importances_, index=treecols).sort_values(by=[0]).tail(30))

lastvalues = train[train.timestamp == 905][['id'] + diff_cols].copy()

verbose = 1
plot = 0
params = {
    'task': 'train',
    'boosting_type': 'dart',
    'objective': 'regression',
    'num_leaves': 1024,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': verbose,
    'max_bin': 512,
    'num_iterations':20
}

X_train, X_test, y_train, y_test = train_test_split(trainer, train.y, test_size=0.2, random_state=0)
train_data = lgb.Dataset(X_train, y_train)
gc.collect()
bst = lgb.train(params, train_data, feval=r_score, verbose_eval=verbose)

df_fi = pd.DataFrame(bst.feature_name(), columns=['feature'])
df_fi['importance'] = list(bst.feature_importance())
df_fi.sort_values('importance', ascending=False, inplace=True)
print(df_fi)
if plot:
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

if verbose:
    y_pred = bst.predict(X_test)
    print('\nScore for another fold: ', r_score(y_test, y_pred))


# train_data = lgb.Dataset(trainer, train.y)
# gc.collect()
# bst = lgb.train(params, train_data, feval=r_score, verbose_eval=verbose)
#
#
# def print_full(x):
#     pd.set_option('display.max_rows', len(x))
#     print(x)
#     pd.reset_option('display.max_rows')
#
# df_fi = pd.DataFrame(bst.feature_name(), columns=['feature'])
# df_fi['importance'] = list(bst.feature_importance())
# df_fi.sort_values('importance', ascending=False, inplace=True)
# print_full(df_fi)

#              feature  importance
# 82      technical_17         122
# 93      technical_30         100
# 103     technical_40          87
# 120   technical_30_d          85
# 119   technical_20_d          82
# 70       technical_2          75
# 85      technical_20          60
# 99      technical_36          52
# 90      technical_27          52
# 139      neighbour16          50
# 148      neighbour25          47
# 96      technical_33          47
# 124       neighbour1          47
# 98      technical_35          46
# 68       technical_0          45
# 84      technical_19          42
# 86      technical_21          41
# 57    fundamental_53          40
# 133      neighbour10          40
# 134      neighbour11          38
# 77      technical_11          38
# 48    fundamental_44          37
# 73       technical_6          36
# 152      neighbour29          35
# 104     technical_41          35
# 74       technical_7          34
# 25    fundamental_21          34
# 71       technical_3          31
# 141      neighbour18          31
# 69       technical_1          31
# 66    fundamental_62          31
# 106     technical_43          29
# 46    fundamental_42          29
# 121   technical_13_d          28
# 33    fundamental_29          27
# 78      technical_12          27
# 45    fundamental_41          27
# 55    fundamental_51          26
# 135      neighbour12          25
# 108          nbnulls          25
# 107     technical_44          25
# 64    fundamental_60          24
# 89      technical_25          24
# 150      neighbour27          24
# 140      neighbour17          24
# 94      technical_31          24
# 41    fundamental_37          23
# 5      fundamental_0          23
# 49    fundamental_45          23
# 51    fundamental_47          22
# 18    fundamental_14          22
# 88      technical_24          22
# 59    fundamental_55          22
# 34    fundamental_30          21
# 128       neighbour5          21
# 79      technical_13          21
# 72       technical_5          20
# 131       neighbour8          20
# 0          derived_0          20
# 23    fundamental_19          20
# 14    fundamental_10          20
# 37    fundamental_33          19
# 15    fundamental_11          19
# 43    fundamental_39          19
# 91      technical_28          18
# 19    fundamental_15          17
# 123       neighbour0          17
# 11     fundamental_7          16
# 138      neighbour15          16
# 4          derived_4          16
# 127       neighbour4          16
# 50    fundamental_46          16
# 22    fundamental_18          16
# 52    fundamental_48          15
# 137      neighbour14          15
# 36    fundamental_32          14
# 38    fundamental_34          14
# 114  technical_44_na          14
# 20    fundamental_16          14
# 53    fundamental_49          14
# 62    fundamental_58          13
# 149      neighbour26          13
# 3          derived_3          13
# 80      technical_14          13
# 26    fundamental_22          13
# 144      neighbour21          13
# 12     fundamental_8          13
# 13     fundamental_9          12
# 27    fundamental_23          12
# 125       neighbour2          12
# 126       neighbour3          11
# 54    fundamental_50          11
# 146      neighbour23          11
# 101     technical_38          11
# 40    fundamental_36          11
# 47    fundamental_43          11
# 39    fundamental_35          10
# 130       neighbour7          10
# 136      neighbour13          10
# 1          derived_1          10
# 21    fundamental_17          10
# 9      fundamental_5           9
# 8      fundamental_3           9
# 143      neighbour20           9
# 2          derived_2           9
# 58    fundamental_54           9
# 63    fundamental_59           9
# 97      technical_34           9
# 17    fundamental_13           8
# 60    fundamental_56           7
# 28    fundamental_24           7
# 24    fundamental_20           7
# 95      technical_32           7
# 100     technical_37           6
# 29    fundamental_25           6
# 102     technical_39           5
# 44    fundamental_40           5
# 35    fundamental_31           5
# 31    fundamental_27           5
# 105     technical_42           5
# 7      fundamental_2           5
# 42    fundamental_38           4
# 56    fundamental_52           4
# 32    fundamental_28           3
# 67    fundamental_63           3
# 76      technical_10           3
# 92      technical_29           2
# 16    fundamental_12           2
# 109   technical_9_na           2
# 147      neighbour24           2
# 75       technical_9           1
# 151      neighbour28           1
# 61    fundamental_57           1
# 115  technical_20_na           1
# 30    fundamental_26           1
# 87      technical_22           1
# 132       neighbour9           1
# 117  technical_13_na           0
# 111  technical_32_na           0
# 118   technical_22_d           0
# 110   technical_0_na           0
# 65    fundamental_61           0
# 145      neighbour22           0
# 112  technical_16_na           0
# 116  technical_30_na           0
# 142      neighbour19           0
# 113  technical_38_na           0
# 6      fundamental_1           0
# 122   technical_34_d           0
# 10     fundamental_6           0
# 81      technical_16           0
# 83      technical_18           0
# 129       neighbour6           0
print("end of training, now predicting")
indice = 0
countplus = 0
rewards = []
while True:
    indice += 1
    test = o.features
    test["nbnulls"] = test.isnull().sum(axis=1)
    if add_nas_ft:
        for elt in nas_cols:
            test[elt + '_na'] = pd.isnull(test[elt]).apply(lambda x: 1 if x else 0)
    test = test.fillna(d_mean)

    pred = o.target
    if add_diff_ft:
        # creating deltas from lastvalues
        indexcommun = list(set(lastvalues.id) & set(test.id))
        lastvalues = pd.concat([test[test.id.isin(indexcommun)]['id'],
                                pd.DataFrame(test[diff_cols][test.id.isin(indexcommun)].values - lastvalues[diff_cols][
                                    lastvalues.id.isin(indexcommun)].values,
                                             columns=diff_cols, index=test[test.id.isin(indexcommun)].index)],
                               axis=1)
        # adding them to test data
        test = test.merge(right=lastvalues, how='left', on='id', suffixes=('', '_d')).fillna(0)
        # storing new lastvalues
        lastvalues = test[['id'] + diff_cols].copy()

    testid = test.id
    test = featureexpander.transform(test[cols])

    # prediction using modelselector and models list
    # selected_prediction = modelselector.predict_proba(test.loc[:, treecols])
    # for ind, elt in enumerate(tokeepmodels):
    #     pred['y'] += selected_prediction[:, ind] * elt.predict(test[tokeepcolumns[ind]])
    pred['y'] = bst.predict(test.loc[:, treecols])

    indexbase = pred.index
    pred.index = testid
    oldpred = pred['y']
    pred.index = indexbase

    o, reward, done, info = env.step(pred)
    rewards.append(reward)
    if reward > 0:
        countplus += 1

    if indice % 100 == 0:
        print(indice, countplus, reward, np.mean(rewards))

    if done:
        print(info["public_score"])
        break
