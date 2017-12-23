import unicodecsv as csv
import codecs
from datetime import datetime
import numpy as np
import pandas as pd
import nltk
import time
import enchant
import re
# function for making ngrams
from nltk.util import ngrams
import collections
import sys
import math
from collections import Counter
from textblob import TextBlob
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import string
import xgboost as xgb
from sklearn import ensemble, linear_model,metrics, model_selection,svm, naive_bayes
from scipy.sparse import hstack
from sklearn.pipeline import  FeatureUnion
from matplotlib.pyplot import  colormaps as cm
from collections import Counter
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
import spacy
from sklearn.model_selection import GridSearchCV


def rmsle(h, y):
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

PATH_TRAIN = './input/train.csv'
PATH_TEST = './input/test.csv'

fl = codecs.open(PATH_TRAIN, 'r','utf-8')
l = fl.readline()

df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")

#
# plt.scatter(x = df_train['number_of_total_atoms'],y = df_train['formation_energy_ev_natom'])
# plt.show()

def get_xyz_data(filename):
    pos_data = []
    lat_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)

idx = df_train.id.values[2]
fn = "input/train/{}/geometry.xyz".format(idx)
train_xyz, train_lat = get_xyz_data(fn)



def runLinRegres(train_X, train_y, test_X, test_y, test_X2):
    model = linear_model.ElasticNetCV(l1_ratio=0.3, eps=0.002)
    model.fit(train_X, train_y)
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2, model

# colormap = plt.cm.viridis
# figure1 = plt.figure(figsize=(19,17))
# plt.title('Corr between data', y=1, size=7)
# sns.heatmap(df_train.astype(float).corr(),linewidths=0.01,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()
#
# colormap = plt.cm.viridis
# figure2 = plt.figure(figsize=(19,17))
# plt.title('Corr between data', y=1, size=7)
# sns.heatmap(df_test.astype(float).corr(),linewidths=0.01,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()



def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=23, child=3, colsample=0.7,params={}):
    param = {}
    param['eta'] = 0.15
    param['objective'] = 'reg:linear'
    param['max_depth'] = 6
    param['silent'] = 1
    param['eval_metric'] = "rmse"
    param['min_child_weight'] = child
    param['subsample'] = 0.6
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 2000

    plst = list(param.items()) + list(params.items())
    xgtrain = xgb.DMatrix(train_X, train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=40)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model

cols_to_drop = ['id','formation_energy_ev_natom','bandgap_energy_ev','percent_atom_in']
train_X = df_train.drop(cols_to_drop, axis=1)
test_X = df_test.drop(['id','percent_atom_in'], axis=1)

kf = model_selection.KFold(n_splits=3, shuffle=True, random_state=10)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([df_train.shape[0]])
params_data = {}
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'
train_y = df_train[target1]

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    #pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0)
    pred_val_y, pred_test_y, model = runLinRegres(dev_X, dev_y, val_X, val_y, test_X)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    print(rmsle(val_y, pred_val_y))
    cv_scores.append(metrics.mean_squared_log_error(val_y, pred_val_y))

plt.scatter(x = pred_train,y = train_y)
plt.
plt.show()


df_test['lin_test_form'] = pred_full_test/.3
df_train['lin_test_form'] = pred_train


test_out = pd.DataFrame()
test_out['id'] = df_test['id']
#test_out = test_out.set_index('id')

#test_out['formation_energy_ev_natom'] = test_out['formation_energy_ev_natom'].apply(lambda x:abs(x))
print('formation_energy_ev_natom',np.mean(cv_scores))
cv_scores.clear()

cols_to_drop = ['id','formation_energy_ev_natom','bandgap_energy_ev','percent_atom_ga']
train_X = df_train.drop(cols_to_drop, axis=1)
test_X = df_test.drop(['id','percent_atom_ga'], axis=1)

pred_full_test = 0
pred_train = np.zeros([df_train.shape[0]])
train_y = df_train[target2]


kf = model_selection.KFold(n_splits=3, shuffle=True, random_state=10)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runLinRegres(dev_X, dev_y, val_X, val_y, test_X)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    print(rmsle(val_y, pred_val_y))
    cv_scores.append(metrics.mean_squared_log_error(val_y, pred_val_y))


df_test['lin_test_band'] = pred_full_test/.3
df_train['lin_test_band'] = pred_train

#------------------------------------------XGboost --------------------------------
cols_to_drop = ['id','formation_energy_ev_natom','bandgap_energy_ev','percent_atom_ga']
train_X = df_train.drop(cols_to_drop, axis=1)
test_X = df_test.drop(['id','percent_atom_ga'], axis=1)
train_y = df_train[target1]

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X,seed_val=23)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    print(rmsle(val_y, pred_val_y))
    cv_scores.append(metrics.mean_squared_log_error(val_y, pred_val_y))
test_out['formation_energy_ev_natom'] = pd.DataFrame(pred_full_test/3.0)
print('formation_energy_ev_natom',np.mean(cv_scores))
cols_to_drop = ['id','formation_energy_ev_natom','bandgap_energy_ev','percent_atom_ga']
train_X = df_train.drop(cols_to_drop, axis=1)
test_X = df_test.drop(['id','percent_atom_ga'], axis=1)
train_y = df_train[target2]
pred_full_test = 0

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X,seed_val=23)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    print(rmsle(val_y, pred_val_y))
    cv_scores.append(metrics.mean_squared_log_error(val_y, pred_val_y))

print('bandgap_energy_ev',np.mean(cv_scores))
test_out['bandgap_energy_ev'] = pd.DataFrame(pred_full_test/3.0)

print(test_out[test_out['bandgap_energy_ev']<0]['bandgap_energy_ev'])
print(test_out[test_out['formation_energy_ev_natom']<0]['formation_energy_ev_natom'])


test_out.to_csv('sub.csv',index=False,float_format='%.4f')