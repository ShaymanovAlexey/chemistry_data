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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from pandas.tools.plotting import table

transformer_exp = FunctionTransformer(np.exp)

def rmsle(h, y):
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

# PATH_TRAIN = './input/train.csv'
# PATH_TEST = './input/test.csv'
#
# fl = codecs.open(PATH_TRAIN, 'r','utf-8')
# l = fl.readline()

df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")
df_sample = pd.read_csv("input/sample_submission.csv")

# print(df_train['percent_atom_al'].sum())
# print(df_train['percent_atom_ga'].sum())
# print(df_train['percent_atom_in'].sum())
#
# fig1, ax1 = plt.subplots()
# ax1.pie([df_train['percent_atom_al'].sum(),df_train['percent_atom_ga'].sum(),df_train['percent_atom_in'].sum()], labels=['percent_atom_al',
#                                                                                                                         'percent_atom_ga',
#                                                                                                                         'percent_atom_in'], autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# plt.show()
#
# print(df_test['percent_atom_al'].sum())
# print(df_test['percent_atom_ga'].sum())
# print(df_test['percent_atom_in'].sum())
#
# fig1, ax1 = plt.subplots()
# ax1.pie([df_test['percent_atom_al'].sum(),df_test['percent_atom_ga'].sum(),df_test['percent_atom_in'].sum()], labels=['percent_atom_al', 'percent_atom_ga',
#                                                                                                                       'percent_atom_in'], autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# plt.show()



# df_train.formation_energy_ev_natom = df_train.formation_energy_ev_natom.round(2)
# df_train.bandgap_energy_ev = df_train.bandgap_energy_ev.round(1)
# cnt_srs_train = df_train['formation_energy_ev_natom'].value_counts()
# #cnt_srs_test = df_test['number_of_total_atoms'].value_counts()
# #
# plt.figure()
# sns.barplot(cnt_srs_train.index, cnt_srs_train.values, alpha=0.9, color='b')
# #sns.barplot(cnt_srs_test.index, cnt_srs_test.values, alpha=0.8,color='g')
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('formation_energy_ev_natom', fontsize=12)
# plt.show()
#



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



def runLinRegres(train_X, train_y, test_X, test_y, test_X2, polyn=True):
    if polyn:
        model = make_pipeline(PolynomialFeatures(2),linear_model.ElasticNetCV(l1_ratio=0.1, eps=0.001))
    else:
        model = linear_model.ElasticNetCV(l1_ratio=0.3, eps=0.00002)
    model.fit(train_X, train_y)
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2, model

def runRandomForest(train_X, train_y, test_X, test_y, test_X2):
    model = ensemble.RandomForestRegressor(n_estimators=300)
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

cols_to_drop = ['id','formation_energy_ev_natom','bandgap_energy_ev']

train_X = df_train.drop(cols_to_drop, axis=1)
test_X = df_test.drop(['id'], axis=1)

kf = model_selection.KFold(n_splits=6, shuffle=True, random_state=4)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([df_train.shape[0],2])
params_data = {}
col = ['formation_energy_ev_natom','bandgap_energy_ev']
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'
train_y = df_train[col]

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y.loc[dev_index], train_y.loc[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0)
    # pred_val_y, pred_test_y, model = runRandomForest(dev_X, dev_y, val_X, val_y, test_X)
    #pred_val_y, pred_test_y, model = runLinRegres(dev_X, dev_y, val_X, val_y, test_X,polyn=False)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    print(rmsle(val_y, pred_val_y))
    cv_scores.append(metrics.mean_squared_log_error(val_y, pred_val_y))

pd_train = pd.DataFrame(pred_train)

print(pd_train[0], train_y)

plt.scatter(x = pd_train[0],y = train_y['formation_energy_ev_natom'])
plt.xlabel('predict_form', fontsize=12)
plt.ylabel('real_form', fontsize=12)
plt.show()


df_test['lin_test_form'] = pred_full_test[:,0]/6.0
df_test['lin_test_band'] = pred_full_test[:,1]/6.0
print(df_test['lin_test_form'])

test_out = pd.DataFrame()
test_out['id'] = df_test['id']
test_out['formation_energy_ev_natom'] = df_test['lin_test_form']
test_out['bandgap_energy_ev'] = df_test['lin_test_band']

plt.scatter(x = pd_train[1],y = train_y['bandgap_energy_ev'])
plt.xlabel('predict_form', fontsize=12)
plt.ylabel('real_form', fontsize=12)
plt.show()

print(test_out[test_out['bandgap_energy_ev']<0]['bandgap_energy_ev'])
#test_out['formation_energy_ev_natom'] = test_out['formation_energy_ev_natom'].apply(lambda x:abs(x))
print(test_out[test_out['formation_energy_ev_natom']<0]['formation_energy_ev_natom'])
test_out.to_csv('sub.csv',index=False,float_format='%.4f')

sys.exit()
#test_out = test_out.set_index('id')

#test_out['formation_energy_ev_natom'] = test_out['formation_energy_ev_natom'].apply(lambda x:abs(x))
print('formation_energy_ev_natom',np.mean(cv_scores))
cv_scores.clear()

cols_to_drop = ['id','formation_energy_ev_natom','bandgap_energy_ev']
train_X = df_train.drop(cols_to_drop, axis=1)
test_X = df_test.drop(['id'], axis=1)

pred_full_test = 0
pred_train = np.zeros([df_train.shape[0]])
del train_y
train_y = df_train[target2]


kf = model_selection.KFold(n_splits=6, shuffle=True, random_state=4)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runLinRegres(dev_X, dev_y, val_X, val_y, test_X)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    print(rmsle(val_y, pred_val_y))
    cv_scores.append(metrics.mean_squared_log_error(val_y, pred_val_y))

plt.scatter(x = pred_train,y = train_y)
plt.xlabel('predict band', fontsize=12)
plt.ylabel('real_band', fontsize=12)
plt.show()


df_test['lin_test_band'] = pred_full_test/6.0
df_train['lin_test_band'] = pred_train/6.0

#------------------------------------------XGboost --------------------------------
# cols_to_drop = ['id','formation_energy_ev_natom','bandgap_energy_ev','lin_test_band']
# train_X = df_train.drop(cols_to_drop, axis=1)
# test_X = df_test.drop(['id','lin_test_band'], axis=1)
# train_y = df_train[target1]
# print("form", train_y)
# kf = model_selection.KFold(n_splits=6, shuffle=True, random_state=10)
# for dev_index, val_index in kf.split(train_X):
#     dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
#     dev_y, val_y = train_y[dev_index], train_y[val_index]
#     pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X,seed_val=23)
#     pred_full_test = pred_full_test + pred_test_y
#     pred_train[val_index] = pred_val_y
#     print(rmsle(val_y, pred_val_y))
#     cv_scores.append(metrics.mean_squared_log_error(val_y, pred_val_y))
test_out['formation_energy_ev_natom'] = df_test['lin_test_form']


# print(test_out['formation_energy_ev_natom'])
# plt.scatter(x = pred_train,y = train_y)
# plt.xlabel('predict form', fontsize=12)
# plt.ylabel('real_form', fontsize=12)
# plt.show()

# print('formation_energy_ev_natom',np.mean(cv_scores))
cols_to_drop = ['id','formation_energy_ev_natom','bandgap_energy_ev','lin_test_form']
train_X = df_train.drop(cols_to_drop, axis=1)
test_X = df_test.drop(['id','lin_test_form'], axis=1)
train_y = df_train[target2]
pred_full_test = 0
kf = model_selection.KFold(n_splits=6, shuffle=True, random_state=10)
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X,seed_val=23)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    print(rmsle(val_y, pred_val_y))
    cv_scores.append(metrics.mean_squared_log_error(val_y, pred_val_y))

plt.scatter(x = pred_train,y = train_y)
plt.xlabel('predict band', fontsize=12)
plt.ylabel('real_band', fontsize=12)
plt.show()


print('bandgap_energy_ev',np.mean(cv_scores))
test_out['bandgap_energy_ev'] = pred_full_test/6.0
#test_out['formation_energy_ev_natom'] = df_sample['formation_energy_ev_natom']
print(test_out[test_out['bandgap_energy_ev']<0]['bandgap_energy_ev'])
#test_out['formation_energy_ev_natom'] = test_out['formation_energy_ev_natom'].apply(lambda x:abs(x))
print(test_out[test_out['formation_energy_ev_natom']<0]['formation_energy_ev_natom'])
test_out.to_csv('sub.csv',index=False,float_format='%.4f')