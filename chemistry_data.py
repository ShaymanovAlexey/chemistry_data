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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgbm
from pandas.tools.plotting import table
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import naive_bayes,neighbors
from sklearn import tree
from catboost import CatBoostRegressor
import itertools
import operator
import atomium
import re

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
#
df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")
# df_sample = pd.read_csv("input/sample_submission.csv")
#
#
# str_name = 'lattice_angle_gamma_degree'
# cnt_srs_train = df_train[str_name].value_counts()
# cnt_srs_test = df_test[str_name].value_counts()
# plt.figure()
# sns.barplot(cnt_srs_train.index, cnt_srs_train.values, alpha=0.9, color='b')
# sns.barplot(cnt_srs_test.index, cnt_srs_test.values, alpha=0.9, color='r')
# #sns.barplot(cnt_srs_test.index, cnt_srs_test.values, alpha=0.8,color='g')
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('Number of: {}'.format(str_name), fontsize=12)
# plt.show()
# #
# cnt_srs_test = df_test[str_name].value_counts()
# plt.figure()
# sns.barplot(cnt_srs_test.index, cnt_srs_test.values, alpha=0.9, color='b')
# #sns.barplot(cnt_srs_test.index, cnt_srs_test.values, alpha=0.8,color='g')
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('Number of: {}'.format(str_name), fontsize=12)
# plt.show()



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

def visualize_bad_occurance(name_pred,y_name,frac_not_fall,list_of_columns):
    train_X["subst_temp"] = train_st_form[name_pred] - train_y[y_name]
    train_X["subst_temp"] = train_X["subst_temp"].apply(lambda x: abs(x))


    for data in list_of_columns:
        temp_train = train_X[train_X["subst_temp"] > frac_not_fall * train_y[y_name]][data].value_counts()
        plt.figure()
        sns.barplot(temp_train.index, temp_train.values, alpha=0.8, color='g')
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel("Date:{0} for {1}".format(data,name_pred), fontsize=12)
        plt.show()
    del train_X["subst_temp"]

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

idx = df_train.id.values[0]
fn = "input/train/{}/geometry.xyz".format(idx)
train_xyz, train_lat = get_xyz_data(fn)

def length(v):
    return np.linalg.norm(v)

def unit_vector(vector):
    return vector / length(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_deg_between(v1, v2):
    return np.degrees(angle_between(v1, v2))

def get_lattice_constants(lattice_vectors):
    lat_const_series = pd.Series()
    for i in range(3):
        str_name = "lattice_vector_"+str(i+1)+"_ang"
        lat_const_series[str_name] = length(lattice_vectors[i])

    lat_const_series["lattice_angle_alpha_degree"] = angle_deg_between(lattice_vectors[1],lattice_vectors[2])
    lat_const_series["lattice_angle_beta_degree"] = angle_deg_between(lattice_vectors[2],lattice_vectors[0])
    lat_const_series["lattice_angle_gamma_degree"] = angle_deg_between(lattice_vectors[0],lattice_vectors[1])
    return lat_const_series

def group_of_data(group_data):
    if group_data < 3:
        return 1
    elif group_data > 2 and group_data < 16:
        return 2
    elif group_data > 15 and group_data < 75:
        return 3
    elif group_data > 74 and group_data < 143:
        return 4
    elif group_data > 142 and group_data < 168:
        return 5
    elif group_data > 167 and group_data < 195:
        return 6
    elif group_data > 194:
        return 7

def get_shortest_distances(reduced_coords):
    natom = len(reduced_coords)
    dist = np.zeros(natom*natom/2)
    for i in range(natom):
        for j in range(i):
            yield  (np.subtract(reduced_coords[i],reduced_coords[natom-1-j]))

def most_common(L):
    dict_d = {}
    L = [length(l) for l in L]
    for l in L:
        if l != 0:
            if dict_d.get(l)==None:
                dict_d[l] =1
            else:
                dict_d[l] +=1
    return max(dict_d.items(), key=operator.itemgetter(1))[0]

df_train["train_xyz"] = df_train['id'].apply(lambda x: get_xyz_data("input/train/{}/geometry.xyz".format(x))[0])
df_train["train_lat"] = df_train['id'].apply(lambda x: get_xyz_data("input/train/{}/geometry.xyz".format(x))[1])
df_train["count_O"] = df_train['train_xyz'].apply(lambda x: len([i[-1] for i in x if i[-1]=='O']))
df_train['count_met'] = df_train['train_xyz'].apply(lambda x: len([i[-1] for i in x if i[-1] !='O']))
df_train['group_data'] = df_train['spacegroup'].apply(lambda x:group_of_data(x))


df_test["train_xyz"] = df_test['id'].apply(lambda x: get_xyz_data("input/test/test/{}/geometry.xyz".format(x))[0])
df_test["train_lat"] = df_test['id'].apply(lambda x: get_xyz_data("input/test/test/{}/geometry.xyz".format(x))[1])
df_test["count_O"] = df_test['train_xyz'].apply(lambda x: len([i[-1] for i in x if i[-1]=='O']))
df_test['count_met'] = df_test['train_xyz'].apply(lambda x: len([i[-1] for i in x if i[-1] !='O']))
df_test['group_data'] = df_test['spacegroup'].apply(lambda x:group_of_data(x))

# for id in df_train['id']:
#     f_open = open("input/train/{}/geometry.xyz".format(id),'r')
#     f_open.readline()
#     str1 = f_open.readline()
#     f_open.readline()
#     f_open.readline()
#     f_open.readline()
#     f_open.readline()
#     data_lines = f_open.readlines()
#     f_open.close()
#
#     fh = open("input/train/{}/geometry_{}.xyz".format(id,'new'), 'w')
#     fh.write(str1)
#     for list_d in data_lines:
#         str = re.search('\w+$',list_d)
#         list_d = str.group(0) + list_d[4:-3]+'\n'
#         fh.write(list_d)
#     fh.close()
#
# for id in df_test['id']:
#     f_open = open("input/test/test/{}/geometry.xyz".format(id),'r')
#     f_open.readline()
#     str1 = f_open.readline()
#     f_open.readline()
#     f_open.readline()
#     f_open.readline()
#     f_open.readline()
#     data_lines = f_open.readlines()
#     f_open.close()
#
#     fh = open("input/test/test/{}/geometry_{}.xyz".format(id,'new'), 'w')
#     fh.write(str1)
#     for list_d in data_lines:
#         str = re.search('\w+$',list_d)
#         list_d = str.group(0) + list_d[4:-3]+'\n'
#         fh.write(list_d)
#     fh.close()
#
# ide_model = atomium.xyz_from_file("input/train/{}/geometry_{}.xyz".format(4,'new'))
# ide_model = ide_model.model()
# print(ide_model.atom(element='Ga')==None)


df_train["model_at"] = df_train['id'].apply(lambda x: atomium.xyz_from_file("input/train/{}/geometry_{}.xyz".format(x,'new')).model())
df_train["mass"]  = df_train["model_at"].apply(lambda x:x.mass())
df_train["gyr_d"]  = df_train["model_at"].apply(lambda x:x.radius_of_gyration())
df_train["mol_ga"]  = df_train["model_at"].apply(lambda x:x.atom(element='Ga'))
df_train["mol_in"]  = df_train["model_at"].apply(lambda x:x.atom(element='In'))
df_train["mol_al"]  = df_train["model_at"].apply(lambda x:x.atom(element='Al'))
df_train["mol_ga_center"]  = df_train.apply(lambda row: row["mol_ga"].distance_to(row["model_at"].center_of_mass()) if row["mol_ga"] !=  None else -1  ,axis=1)
df_train["mol_in_center"]  = df_train.apply(lambda row: row["mol_in"].distance_to(row["model_at"].center_of_mass()) if row["mol_in"] !=  None else -1  ,axis=1)
df_train["mol_al_center"]  = df_train.apply(lambda row: row["mol_al"].distance_to(row["model_at"].center_of_mass()) if row["mol_al"] !=  None else -1 ,axis=1)
# df_train["pair_atoms"]  = df_train["model_at"].apply(lambda x:list(x.pairwise_atoms()))
# ide_model = atomium.xyz_from_file("input/train/{}/geometry_{}.xyz".format(8,'new'))
# ide_model = ide_model.model()
# print(list(ide_model.pairwise_atoms())[0][0])


df_test["model_at"] = df_test['id'].apply(lambda x: atomium.xyz_from_file("input/train/{}/geometry_{}.xyz".format(20,'new')).model())
df_test["mass"]  = df_test["model_at"].apply(lambda x:x.mass())
df_test["gyr_d"]  = df_test["model_at"].apply(lambda x:x.radius_of_gyration())
df_test["mol_ga"]  = df_test["model_at"].apply(lambda x:x.atom(element='Ga'))
df_test["mol_in"]  = df_test["model_at"].apply(lambda x:x.atom(element='In'))
df_test["mol_al"]  = df_test["model_at"].apply(lambda x:x.atom(element='Al'))
df_test["mol_ga_center"]  = df_test.apply(lambda row: row["mol_ga"].distance_to(row["model_at"].center_of_mass()) if row["mol_ga"] !=  None else -1  ,axis=1)
df_test["mol_in_center"]  = df_test.apply(lambda row: row["mol_in"].distance_to(row["model_at"].center_of_mass()) if row["mol_in"] !=  None else -1  ,axis=1)
df_test["mol_al_center"]  = df_test.apply(lambda row: row["mol_al"].distance_to(row["model_at"].center_of_mass()) if row["mol_al"] !=  None else -1  ,axis=1)
# df_test["pair_atoms"]  = df_test["model_at"].apply(lambda x:list(x.pairwise_atoms()))
# ide_model = atomium.xyz_from_file("input/train/{}/geometry_{}.xyz".format(20,'new'))
# ide_model = ide_model.model()
# print(len(list(ide_model.pairwise_atoms())))

for i in range(3):
    df_train["mass_" + str(i)] = df_train["model_at"].apply(lambda x: x.center_of_mass()[i])
    df_test["mass_" + str(i)] = df_test["model_at"].apply(lambda x: x.center_of_mass()[i])

min_atoms = 10

for i in range(min_atoms):
    # df_train["pair_atoms"+"_"+str(i)] = df_train["pair_atoms"].apply(lambda x: x[i][1])
    # df_test["pair_atoms"+"_"+str(i)] = df_test["pair_atoms"].apply(lambda x: x[i][1])
    # df_train["center_to_atoms" + "_" + str(i)] = df_train.apply(lambda row: row["pair_atoms"+"_"+str(i)].distance_to(row["model_at"].center_of_mass()),axis=1)
    # df_test["center_to_atoms" + "_" + str(i)] = df_test.apply(lambda row: row["pair_atoms"+"_"+str(i)].distance_to(row["model_at"].center_of_mass()),axis=1)
    df_train["positional_vector_"+str(i)] = df_train['train_xyz'].apply(lambda x:x[i][0])
    df_test["positional_vector_"+str(i)] = df_test['train_xyz'].apply(lambda x:x[i][0])
#
for i in range(min_atoms):
    for j in range(3):
         df_train["positional_vector_"+str(i)+"_"+str(j)] = df_train["positional_vector_"+str(i)].apply(lambda x: x[j])
         df_test["positional_vector_" + str(i)+"_"+str(j)] = df_test["positional_vector_"+str(i)].apply(lambda x: x[j])
    # for j in range(3):
    # df_train["positional_vector_l_"+str(i)] = df_train["positional_vector_"+str(i)].apply(lambda x: length(x))
    # df_test["positional_vector_l_" + str(i)] = df_test["positional_vector_"+str(i)].apply(lambda x: length(x))

# df_train['lattice_angle_gamma_degree'] = df_train['lattice_angle_gamma_degree'].round(1)
# cnt_srs_train = df_train['number_of_total_atoms'].value_counts()
#
#
# plt.figure()
# sns.barplot(cnt_srs_train.index, cnt_srs_train.values, alpha=0.9, color='b')
# #sns.barplot(cnt_srs_test.index, cnt_srs_test.values, alpha=0.8,color='g')
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('round_band_gap', fontsize=12)
# plt.show()

df_train["volume"] = df_train['train_lat'].apply(lambda x:np.linalg.det(np.transpose(x)))
df_test["volume"] = df_test['train_lat'].apply(lambda x:np.linalg.det(np.transpose(x)))
#
df_train["den_met"] = df_train.apply(lambda row: row['count_met']/row['volume'],axis=1)
df_test["den_met"] = df_test.apply(lambda row: row['count_met']/row['volume'],axis=1)

df_train["reciprocal_lattice_vectors"] = df_train['train_lat'].apply(lambda x:np.linalg.inv(np.transpose(x)))
df_test["reciprocal_lattice_vectors"] = df_test['train_lat'].apply(lambda x:np.linalg.inv(np.transpose(x)))

# get all reduced vectors
df_train['all_red_vectors'] = df_train.apply(lambda row:[[np.matmul(row['reciprocal_lattice_vectors'], R)] for (R, _) in row['train_xyz']],axis=1)
# df_train['rad_vectors'] = df_train['all_red_vectors'].apply(lambda x: list(get_shortest_distances(x)))
# df_train['most_rad_vectors'] = df_train['rad_vectors'].apply(lambda x: most_common(x))


df_test['all_red_vectors'] = df_test.apply(lambda row:[[np.matmul(row['reciprocal_lattice_vectors'], R)] for (R, _) in row['train_xyz']],axis=1)
# df_test['rad_vectors'] = df_test['all_red_vectors'].apply(lambda x: list(get_shortest_distances(x)))
# df_test['most_rad_vectors'] = df_test['rad_vectors'].apply(lambda x: most_common(x))

# for i in range(3):
#     for j in range(3):
#         df_train['mean_rad_vectors_'+str(i)+'_'+str(j)] = df_train['rad_vectors'].apply(lambda x: np.mean(x,axis=1)[i][j])
#         df_train['min_rad_vectors_'+str(i)+'_'+str(j)] = df_train['rad_vectors'].apply(lambda x: np.min(x,axis=1)[i][j])
#         df_train['max_rad_vectors_' + str(i) + '_' + str(j)] = df_train['rad_vectors'].apply(lambda x: np.max(x, axis=1)[i][j])
#         df_test['mean_rad_vectors_' + str(i)+'_'+str(j)] = df_test['rad_vectors'].apply(lambda x: np.mean(x, axis=1)[i][j])
#         df_test['min_rad_vectors_' + str(i)+'_'+str(j)] = df_test['rad_vectors'].apply(lambda x: np.min(x, axis=1)[i][j])
#         df_test['max_rad_vectors_' + str(i) + '_' + str(j)] = df_test['rad_vectors'].apply(lambda x: np.max(x, axis=1)[i][j])



for i in range(min_atoms):
    # for j in range(3):
    df_train["reduced_coordinate_vector_" + str(i)] = df_train.apply(lambda row: [np.matmul(row['reciprocal_lattice_vectors'], row['positional_vector_' + str(i)])], axis=1)
    df_test["reduced_coordinate_vector_" + str(i)] = df_test.apply(lambda row: [np.matmul(row['reciprocal_lattice_vectors'], row['positional_vector_'+str(i)])], axis=1)
    df_train["reduced_coordinate_vector_" + str(i)] = df_train["reduced_coordinate_vector_" + str(i)].apply(lambda row: length(row))
    df_test["reduced_coordinate_vector_" + str(i)] = df_test["reduced_coordinate_vector_" + str(i)].apply(lambda row: length(row))

    # df_train["reduced_coordinate_vector_"+str(i)+"_"+str(j)] = df_train.apply(lambda row: np.matmul(row['reciprocal_lattice_vectors'],row['positional_vector_'+str(i)])[j],axis=1)
        # df_test["reduced_coordinate_vector_" + str(i)+"_"+str(j)] = df_test.apply(lambda row: np.matmul(row['reciprocal_lattice_vectors'], row['positional_vector_'+str(i)])[j], axis=1)

for i in range(min_atoms):
    del df_train["positional_vector_"+str(i)],df_test["positional_vector_"+str(i)]
    # del df_train["pair_atoms" + "_" + str(i)],df_test["pair_atoms" + "_" + str(i)]
del df_test["train_xyz"],df_train["train_xyz"],df_test["train_lat"],df_train["train_lat"]
del df_train["reciprocal_lattice_vectors"],df_test['reciprocal_lattice_vectors']
del df_train['all_red_vectors'],df_test['all_red_vectors']
del df_train["mol_ga"],df_test["mol_ga"]
del df_train["mol_in"],df_test["mol_in"]
del df_train["mol_al"],df_test["mol_al"]

# del df_train['rad_vectors'],df_test['rad_vectors']

# del df_train["pair_atoms"], df_test["pair_atoms"]
del df_train["model_at"], df_test["model_at"]

def run_model(model,train_X, train_y, test_X, test_y, test_X2, polyn):
    if polyn:
        model = make_pipeline(PolynomialFeatures(2),model)
    else:
        model = model
    model.fit(train_X, train_y)
    pred_test_y = model.predict(test_X)
    pred_test_y2 = model.predict(test_X2)
    return pred_test_y, pred_test_y2, model

def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=23, child=3,params={}):
    param = {}
    param['eta'] = 0.012
    param['objective'] = 'reg:linear'
    param['max_depth'] = 4
    param['silent'] = 1
    param['eval_metric'] = "rmse"
    param['min_child_weight'] = child
    param['subsample'] = 0.79
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = 1000
    #col = ['formation_energy_ev_natom', 'bandgap_energy_ev']

    plst = list(param.items()) + list(params.items())
    xgtrain = xgb.DMatrix(train_X,train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, test_y)
        watchlist = [(xgtrain,'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=150, verbose_eval=2000)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y1 = model.predict(xgtest, ntree_limit = model.best_ntree_limit)

    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)

    pred_train = np.zeros([test_X.shape[0]])
    pred_train = pred_test_y1
    pred_test = np.zeros([test_X2.shape[0]])

    if test_X2 is not None:
        pred_test = pred_test_y2
    return pred_train, pred_test, model

def runLGBM(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=23, child=3, rounds = 600):
    RS = 20170501

    ROUNDS = rounds

    params = {
        'objective': 'regression_l2',
        'metric': 'MSE',
        'boosting': 'gbdt',
        'learning_rate': 0.03,
        'verbose': 0,
        'num_leaves': 2 ** 5,
        'bagging_fraction': 0.95,
        'bagging_freq': 3,
        'bagging_seed': RS,
        'feature_fraction': 0.7,
        'feature_fraction_seed': RS,
        'max_bin': 100,
        'max_depth': 3,
        'num_rounds': ROUNDS
    }

    train_lgb = lgbm.Dataset(train_X, train_y)


    test_lgbm = test_X
    model = lgbm.train(params, train_lgb, num_boost_round=ROUNDS)

    pred_test_y1 = model.predict(test_lgbm)

    if test_X2 is not None:
        test_lgbm2 = test_X2
        pred_test_y2 = model.predict(test_lgbm2)

    pred_train = np.zeros([test_X.shape[0]])
    pred_train = pred_test_y1
    pred_test = np.zeros([test_X2.shape[0]])

    if test_X2 is not None:
        pred_test = pred_test_y2
    return pred_train, pred_test, model

def run_estimator(k_enum,run_for_model,train_X, train_y, test_X, pol=False,single=False,params_xgb={}, round_data=600):
    rmsle_data = []
    pred_full_test = 0
    if single:
        pred_train = np.zeros([train_X.shape[0]])
    else:
        k_data = train_y.shape[1]
        pred_train = np.zeros([train_X.shape[0],k_data])
    params_data = {}
    RS = 20180121
    r_d = 2021

    kf = model_selection.KFold(n_splits=k_enum, shuffle=True, random_state=r_d)
    cv_score = []
    pred_full_test = 0

    for dev_index, val_index in kf.split(train_X):
        dev_X, val_X = train_X.iloc[dev_index], train_X.iloc[val_index]
        dev_y, val_y = train_y.iloc[dev_index], train_y.iloc[val_index]
        if run_for_model== runXGB:
            print("runXGB")
            pred_val_y, pred_test_y, model1= runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0,params=params_xgb)
        elif run_for_model==runLGBM:
            print("runLGBM")
            pred_val_y, pred_test_y, model1 = runLGBM(dev_X, dev_y, val_X, val_y, test_X, seed_val=0,rounds=round_data)
        else:
            pred_val_y, pred_test_y, model = run_model(run_for_model,dev_X, dev_y, val_X, val_y, test_X,pol)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        # print("rmsle",rmsle(val_y, pred_val_y))
        rmsle_data.append(rmsle(val_y, pred_val_y))
        #cv_scores.append(metrics.mean_squared_log_error(val_y, pred_val_y))

    pred_full_test = pred_full_test / float(k_enum)
    # pred_full_test[pred_full_test <= 0] = 1e-6
    return pred_train,pred_full_test

def run_m_classes(k_enum, clrf_list, clr_names, train_X, train_y, test_X):
    for clr, name in zip(clrf_list, clr_names):
        scores = model_selection.cross_val_score(clr,train_X,train_y, cv = k_enum)
        y_pred = model_selection.cross_val_predict(clr,train_X,train_y,cv = k_enum)
        print(rmsle(y_pred,train_y))
        print("Accuracy: %0.2f +/- %0.2f (%s 95%% CI)" % (scores.mean(), scores.std() * 2, name))

#-----------------------------------------round band gap for classification task ------------------------------------------------------------------
# df_train.formation_energy_ev_natom = df_train.formation_energy_ev_natom.round(2)
# df_round_band_gap = df_train.copy()
# df_round_band_gap['round_band_gap'] = df_train.bandgap_energy_ev.round(0)
# # df_round_band_gap['round_band_gap'] = df_round_band_gap['round_band_gap'].apply(lambda x:int(x))
# cnt_srs_train = df_round_band_gap['round_band_gap'].value_counts()
# df_y = df_round_band_gap['round_band_gap']
# df_round_band_gap = df_round_band_gap.drop(['id','formation_energy_ev_natom','round_band_gap','bandgap_energy_ev'], axis=1)
#
# df_test_round_band_gap = df_test.copy()
# df_test_round_band_gap = df_test_round_band_gap.drop('id', axis =1)
# cnt_srs_test = df_test['number_of_total_atoms'].value_counts()
#
# plt.figure()
# sns.barplot(cnt_srs_train.index, cnt_srs_train.values, alpha=0.9, color='b')
# #sns.barplot(cnt_srs_test.index, cnt_srs_test.values, alpha=0.8,color='g')
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('round_band_gap', fontsize=12)
# plt.show()

train_st_form = pd.DataFrame()
test_st_form = pd.DataFrame()

train_st_band = pd.DataFrame()
test_st_band = pd.DataFrame()


# k_enum = 5
# clf1 = ensemble.AdaBoostClassifier()
# clf2 = ensemble.RandomForestClassifier(n_estimators=350,random_state=1)
# clf3 = naive_bayes.GaussianNB()
# clf4 = neighbors.KNeighborsClassifier()
# clf5 = tree.DecisionTreeClassifier()
# clf6 = ensemble.GradientBoostingClassifier(n_estimators=300,random_state=201)
# clf7 = svm.NuSVR(C=6)
# clrf_list = [clf1,clf2,clf3,clf4,clf5,clf6,clf7]
# clr_names = ['AdaBoostClassifier','RandomForestClassifier','GaussianNB','KNeighborsClassifier','DecisionTreeClassifier','GradientBoostingClassifier','NuSVR']
# run_m_classes(k_enum, clrf_list, clr_names, df_round_band_gap, df_y, df_test_round_band_gap)
#
# eclf = VotingClassifier(estimators=[('cl1', clf1),('cl2', clf2),('cl3',clf3),('cl4', clf4), ('cl5', clf5),('cl6',clf6)], voting='soft', weights=[1,2,1,2,1,3], flatten_transform=True)
# clr_train,clr_test = run_estimator(k_enum,eclf,df_round_band_gap, df_y, df_test_round_band_gap,False,True)
# print(metrics.accuracy_score(clr_train,df_y))
#
# #---------------------------------------------------------------------------------------------------------------------------------------------------
# train_st_band["round_band"] = clr_train
# test_st_band["round_band"] = clr_test


# colormap = plt.cm.viridis
# figure1 = plt.figure(figsize=(19,17))
# plt.title('Corr between data', y=1, size=7)
# sns.heatmap(df_train.astype(float).corr(),linewidths=0.01,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()



cols_to_drop = ['id','formation_energy_ev_natom','bandgap_energy_ev']
params = None
col = ['formation_energy_ev_natom','bandgap_energy_ev']
train_X = df_train.drop(cols_to_drop, axis=1)
test_X = df_test.drop(['id'], axis=1)
# test_X_good = df_test[df_test.spacegroup != 12]
# test_X_good = test_X_good.drop(['id'], axis=1)
# train_X_good = df_train[df_train.spacegroup != 12]
# train_y_good = train_X_good[col]
# train_X_good = train_X_good.drop(cols_to_drop, axis=1)

# print(pd.isnull(train_y_good).any(1).nonzero())
#col = ['bandgap_energy_ev']
train_y = df_train[col]
k_enum=9
list_of_columns = ['spacegroup',
                   'number_of_total_atoms',
                   'percent_atom_al',
                   'percent_atom_ga',
                   'percent_atom_in']
print("CatBoost")

cat_form_train3,cat_form_test3 = run_estimator(k_enum,CatBoostRegressor(random_seed=101,iterations=800,depth=2, l2_leaf_reg=3,learning_rate=0.11, loss_function='RMSE',feature_border_type='MinEntropy'),train_X, train_y['formation_energy_ev_natom'], test_X,False,True)
cat_band_train3,cat_band_test3 = run_estimator(k_enum,CatBoostRegressor(random_seed=101,iterations=800,depth=6, l2_leaf_reg=3, learning_rate=0.06, loss_function='RMSE'),train_X, train_y['bandgap_energy_ev'], test_X,False,True)

train_st_form['cat_form2'] = cat_form_train3
train_st_band['cat_band2'] = cat_band_train3
test_st_form['cat_form2'] = cat_form_test3
test_st_band['cat_band2'] = cat_band_test3
print("cat form ",rmsle(cat_form_train3,train_y['formation_energy_ev_natom']))
print("cat  band ",rmsle(cat_band_train3,train_y['bandgap_energy_ev']))

list_of_columns = ['spacegroup',
                   'number_of_total_atoms',
                   'percent_atom_al',
                   'percent_atom_ga',
                   'percent_atom_in']

# visualize_bad_occurance('cat_form2','formation_energy_ev_natom',1/3,list_of_columns)

# print("nuSVR")
# nus_form_train3,nus_form_test3 = run_estimator(k_enum,svm.NuSVR(C=3),train_X, train_y['formation_energy_ev_natom'], test_X,False,True)
# nus_band_train3,nus_band_test3 = run_estimator(k_enum,svm.NuSVR(C=100),train_X, train_y['bandgap_energy_ev'], test_X,False,True)
#
# train_st_form['nuSVR_form2'] = nus_form_train3
# train_st_band['nuSVR_band2'] = nus_band_train3
# test_st_form['nuSVR_form2'] = nus_form_test3
# test_st_band['nuSVR_band2'] = nus_band_test3
# print("nuSVR form ",rmsle(nus_form_train3,train_y['formation_energy_ev_natom']))
# print("nuSVR band ",rmsle(nus_band_train3,train_y['bandgap_energy_ev']))

# visualize_bad_occurance('nuSVR_form2','formation_energy_ev_natom',1/3,list_of_columns)

# subPd = pd.DataFrame(nus_form_train3,train_y['formation_energy_ev_natom'])
# print(subPd[abs(subPd['formation_energy_ev_natom']- subPd[0])>1/3*subPd['formation_energy_ev_natom']])
# plt.scatter(nus_form_train3,train_y['formation_energy_ev_natom'])
# plt.xlabel('predict data')
# plt.ylabel('true data')
# plt.show()

print("Knn")

knn_train,knn_test = run_estimator(k_enum,neighbors.KNeighborsRegressor(),train_X, train_y, test_X,False)

train_st_form['knn_form'] = knn_train[:,0]
train_st_band['knn_band'] = knn_train[:,1]
test_st_form['knn_form'] = knn_test[:,0]
test_st_band['knn_band'] = knn_test[:,1]
print("knn form ",rmsle(knn_train[:,0],train_y['formation_energy_ev_natom']))
print("knn band ",rmsle(knn_train[:,1],train_y['bandgap_energy_ev']))

# visualize_bad_occurance('knn_form','formation_energy_ev_natom',1/3,list_of_columns)
#
#
print("LinReg")
lin_train,lin_test = run_estimator(k_enum,linear_model.LinearRegression(),train_X, train_y, test_X,True)

train_st_form['lin_form'] = lin_train[:,0]
train_st_band['lin_band'] = lin_train[:,1]
test_st_form['lin_form'] = lin_test[:,0]
test_st_band['lin_band'] = lin_test[:,1]
print("reg form ",rmsle(lin_train[:,0],train_y['formation_energy_ev_natom']))
print("reg band ",rmsle(lin_train[:,1],train_y['bandgap_energy_ev']))#
# visualize_bad_occurance('lin_form','formation_energy_ev_natom',1/3,list_of_columns)

print("XGB")

# best_form = {}
# best_band = {}
# for i in [x/100. for x in range(60,80,2)]:
xgb_train_form,xgb_test_form = run_estimator(k_enum,runXGB,train_X, train_y['formation_energy_ev_natom'], test_X,False,True, params_xgb={'eta':1,'subsample':0.74,'colsample_bytree':0.72})
    # best_form[rmsle(xgb_train_form, train_y['formation_energy_ev_natom'])]=i

# for i in [x/100. for x in range(40,60,2)]:
xgb_train_band,xgb_test_band = run_estimator(k_enum,runXGB,train_X, train_y['bandgap_energy_ev'], test_X,False,True,params_xgb={'eta':0.047,'subsample':0.11,'colsample_bytree':0.54})
    # best_band[rmsle(xgb_train_band, train_y['bandgap_energy_ev'])] = i

#
# print("form best {0} for {1} ".format(min(best_form.keys()),best_form[min(best_form.keys())]))
# print("band best {0} for {1} ".format(min(best_band.keys()),best_band[min(best_band.keys())]))


train_st_form['xgb_form'] = xgb_train_form
train_st_band['xgb_band'] = xgb_train_band
test_st_form['xgb_form'] = xgb_test_form
test_st_band['xgb_band'] = xgb_test_band
print("xgb form ",rmsle(xgb_train_form,train_y['formation_energy_ev_natom']))
print("xgb band ",rmsle(xgb_train_band,train_y['bandgap_energy_ev']))

print("RandomForest")
r_tree_train,r_tree_test=run_estimator(k_enum,ensemble.RandomForestRegressor(n_estimators=300,random_state=10),train_X, train_y, test_X)

# light_train_form_good,light_test_form_good = run_estimator(k_enum,ensemble.RandomForestRegressor(n_estimators=300,random_state=10),train_X_good, train_y_good, test_X_good)

train_st_form['rand_form'] = r_tree_train[:,0]
train_st_band['rand_band'] = r_tree_train[:,1]
test_st_form['rand_form'] = r_tree_test[:,0]
test_st_band['rand_band'] = r_tree_test[:,1]
print("RandomForest form ",rmsle(r_tree_train[:,0],train_y['formation_energy_ev_natom']))
# print("RandomForest form_good ",rmsle(light_train_form_good,train_y_good))
print("RandomForest band ",rmsle(r_tree_train[:,1],train_y['bandgap_energy_ev']))


print("LighGBM")

# best_form = {}
# best_band = {}
# for i in [x/100. for x in range(60,80,2)]:
light_train_form,light_test_form = run_estimator(k_enum,runLGBM,train_X.drop(["count_O","count_met"],axis=1),
                                                 train_y['formation_energy_ev_natom'],
                                                 test_X.drop(["count_O","count_met"],axis=1),False,True,round_data=600)

# light_train_form_good,light_test_form_good = run_estimator(k_enum,runLGBM,train_X_good, train_y_good['formation_energy_ev_natom'], test_X_good ,False,True)
    # best_form[rmsle(xgb_train_form, train_y['formation_energy_ev_natom'])]=i

# for i in [x/100. for x in range(40,60,2)]:
light_train_band,light_test_band = run_estimator(k_enum,runLGBM,train_X,
                                                 train_y['bandgap_energy_ev'],
                                                 test_X,False,True,round_data=550)

train_st_form['light_form'] = light_train_form
train_st_band['light_band'] = light_train_band
test_st_form['light_form'] = light_test_form
test_st_band['light_band'] = light_test_band
print("light_form ",rmsle(light_train_form,train_y['formation_energy_ev_natom']))
print("light_band ",rmsle(light_train_band,train_y['bandgap_energy_ev']))
# print("light_form_good ",rmsle(light_train_form_good,train_y_good['formation_energy_ev_natom']))

# visualize_bad_occurance('light_form','formation_energy_ev_natom',1/3,list_of_columns)



# #
#
# print("XGB")
#
# # best_form = {}
# # best_band = {}
# # for i in [x/100. for x in range(60,80,2)]:
# xgb_train_form,xgb_test_form = run_estimator(k_enum,runXGB,train_X, train_y['formation_energy_ev_natom'], test_X,False,True, params_xgb={'eta':1,'subsample':0.74,'colsample_bytree':0.72})
#     # best_form[rmsle(xgb_train_form, train_y['formation_energy_ev_natom'])]=i
#
# # for i in [x/100. for x in range(40,60,2)]:
# xgb_train_band,xgb_test_band = run_estimator(k_enum,runXGB,train_X, train_y['bandgap_energy_ev'], test_X,False,True,params_xgb={'eta':0.047,'subsample':0.11,'colsample_bytree':0.54})
#     # best_band[rmsle(xgb_train_band, train_y['bandgap_energy_ev'])] = i
#
# #
# # print("form best {0} for {1} ".format(min(best_form.keys()),best_form[min(best_form.keys())]))
# # print("band best {0} for {1} ".format(min(best_band.keys()),best_band[min(best_band.keys())]))
#
#
# train_st_form['xgb_form'] = xgb_train_form
# train_st_band['xgb_band'] = xgb_train_band
# test_st_form['xgb_form'] = xgb_test_form
# test_st_band['xgb_band'] = xgb_test_band
# print("xgb form ",rmsle(xgb_train_form,train_y['formation_energy_ev_natom']))
# print("xgb band ",rmsle(xgb_train_band,train_y['bandgap_energy_ev']))
# visualize_bad_occurance('xgb_form','formation_energy_ev_natom',1/3,list_of_columns)



# print("RandomForest")
# r_tree_train,r_tree_test=run_estimator(k_enum+1,ensemble.RandomForestRegressor(n_estimators=300,random_state=10),train_X, train_y, test_X)
#
# # light_train_form_good,light_test_form_good = run_estimator(k_enum,ensemble.RandomForestRegressor(n_estimators=300,random_state=10),train_X_good, train_y_good, test_X_good)
#
# train_st_form['rand_form'] = r_tree_train[:,0]
# train_st_band['rand_band'] = r_tree_train[:,1]
# test_st_form['rand_form'] = r_tree_test[:,0]
# test_st_band['rand_band'] = r_tree_test[:,1]
# print("RandomForest form ",rmsle(r_tree_train[:,0],train_y['formation_energy_ev_natom']))
# # print("RandomForest form_good ",rmsle(light_train_form_good,train_y_good))
# print("RandomForest band ",rmsle(r_tree_train[:,1],train_y['bandgap_energy_ev']))
# # visualize_bad_occurance('rand_form','formation_energy_ev_natom',1/3,list_of_columns)
# print("RandomForestExtra")
# r_tree_train2,r_tree_test2=run_estimator(k_enum-1,ensemble.RandomForestRegressor(n_estimators=600,random_state=10),train_X, train_y, test_X)
#
# train_st_form['rand_form2'] = r_tree_train2[:,0]
# train_st_band['rand_band2'] = r_tree_train2[:,1]
# test_st_form['rand_form2'] = r_tree_test2[:,0]
# test_st_band['rand_band2'] = r_tree_test2[:,1]




print("ExtraTreeRegressor")
e_tree_train,e_tree_test  = run_estimator(k_enum,ensemble.ExtraTreesRegressor(n_estimators=500),train_X, train_y, test_X)

train_st_form['e_tree_form'] = e_tree_train[:,0]
train_st_band['e_tree_band'] = e_tree_train[:,1]
test_st_form['e_tree_form'] = e_tree_test[:,0]
test_st_band['e_tree_band'] = e_tree_test[:,1]
# visualize_bad_occurance('e_tree_form', 'formation_energy_ev_natom', 1 / 3, list_of_columns)


# light_train_form_new,light_test_form_new = run_estimator(k_enum,runLGBM,train_st_form, train_y['formation_energy_ev_natom'], test_st_form,False,True)
#
# light_train_band_new,light_test_band_new = run_estimator(k_enum,runLGBM,train_st_band, train_y['bandgap_energy_ev'], test_st_band,False,True)
#
#
# print("light_new form ",rmsle(light_train_form_new,train_y['formation_energy_ev_natom']))
# print("light_new band ",rmsle(light_train_band_new,train_y['bandgap_energy_ev']))
#
# plt.scatter(light_train_form_new,train_y['formation_energy_ev_natom'])
# plt.xlabel('predict data')
# plt.ylabel('true data')
# plt.show()
# test_out = pd.DataFrame()
# test_out['id'] = df_test['id']
# test_out['formation_energy_ev_natom'] = test_st_form.mean(axis=1)
# test_out['bandgap_energy_ev'] = test_st_band.mean(axis=1)
#
# print(test_out[test_out['bandgap_energy_ev']<0]['bandgap_energy_ev'])
# test_out['formation_energy_ev_natom'] = test_out['formation_energy_ev_natom'].apply(lambda x:abs(x))
# print(test_out[test_out['formation_energy_ev_natom']<0]['formation_energy_ev_natom'])
# test_out.to_csv('sub.csv',index=False)
#





# print("Lin new Regressor")
# lin_train_form_st,lin_test_form_st  = run_estimator(k_enum,linear_model.LinearRegression(),train_st_form, train_y['formation_energy_ev_natom'], test_st_form,True,True)
# lin_train_band_st,lin_test_band_st  = run_estimator(k_enum,linear_model.LinearRegression(),train_st_band, train_y['bandgap_energy_ev'], test_st_band,True,True)
# print("Lin new Regressor form",rmsle(lin_train_form_st,train_y['formation_energy_ev_natom']))
# print("Lin new Regressor  band ",rmsle(lin_train_band_st,train_y['bandgap_energy_ev']))
#
#
# print("ExtraTreeRegressor")
# e_tree_train_form_st,e_tree_test_form_st  = run_estimator(k_enum,ensemble.ExtraTreesRegressor(n_estimators=300),train_st_form, train_y['formation_energy_ev_natom'], test_st_form,False,True)
# e_tree_train_band_st,e_tree_test_band_st  = run_estimator(k_enum,ensemble.ExtraTreesRegressor(n_estimators=300),train_st_band, train_y['bandgap_energy_ev'], test_st_band,False,True)
# print("ExtraTreeRegressor form ",rmsle(e_tree_train_form_st,train_y['formation_energy_ev_natom']))
# print("ExtraTreeRegressor band ",rmsle(e_tree_train_band_st,train_y['bandgap_energy_ev']))


print("XGB_new")
xgb_train_form_st,xgb_test_form_st = run_estimator(k_enum,runXGB,train_st_form, train_y['formation_energy_ev_natom'], test_st_form,False,True,params_xgb={'eta':0.008,'subsample':0.74,'colsample_bytree':0.72})
xgb_train_band_st,xgb_test_band_st = run_estimator(k_enum,runXGB,train_st_band, train_y['bandgap_energy_ev'], test_st_band,False,True,params_xgb={'eta':0.047,'subsample':0.11,'colsample_bytree':0.54})
print("XGB_new form ",rmsle(xgb_train_form_st,train_y['formation_energy_ev_natom']))
print("XGB_new band ",rmsle(xgb_train_band_st,train_y['bandgap_energy_ev']))

test_out = pd.DataFrame()
test_out['id'] = df_test['id']
test_out['formation_energy_ev_natom'] = xgb_test_form_st
test_out['bandgap_energy_ev'] = xgb_test_band_st

print(test_out[test_out['bandgap_energy_ev']<0]['bandgap_energy_ev'])
print(test_out[test_out['formation_energy_ev_natom']<0]['formation_energy_ev_natom'])
test_out['formation_energy_ev_natom'] = test_out['formation_energy_ev_natom'].apply(lambda x:abs(x))
test_out.to_csv('sub.csv',index=False)


# print("NuSVU new")
# NuSVU_train_form_st,NuSVU_test_form_st = run_estimator(k_enum,svm.NuSVR(C=120),train_st_form, train_y['formation_energy_ev_natom'], test_st_form,False,True)
# NuSVU_train_band_st,NuSVU_test_band_st = run_estimator(k_enum,svm.NuSVR(C=0.5),train_st_band, train_y['bandgap_energy_ev'], test_st_band,False,True)
# print("NuSVU new form ",rmsle(NuSVU_train_form_st,train_y['formation_energy_ev_natom']))
# print("NuSVU new band ",rmsle(NuSVU_train_band_st,train_y['bandgap_energy_ev']))
#
# print("lightGDB new")
# light_train_form_st,light_test_form_st = run_estimator(k_enum,runLGBM,train_st_form,
#                                                  train_y['formation_energy_ev_natom'],
#                                                        test_st_form,False,True,round_data=600)
#
# # light_train_form_good,light_test_form_good = run_estimator(k_enum,runLGBM,train_X_good, train_y_good['formation_energy_ev_natom'], test_X_good ,False,True)
#     # best_form[rmsle(xgb_train_form, train_y['formation_energy_ev_natom'])]=i
#
# # for i in [x/100. for x in range(40,60,2)]:
# light_train_band_st,light_test_band_st = run_estimator(k_enum,runLGBM,train_st_form,
#                                                  train_y['bandgap_energy_ev'],
#                                                        test_st_form,False,True,round_data=550)
#
# print("lightGDB form ",rmsle(light_train_form_st,train_y['formation_energy_ev_natom']))
# print("lightGDB band ",rmsle(light_train_band_st,train_y['bandgap_energy_ev']))
#

#print(xgb_test_form,xgb_test_band)

# print("ExtraTreeRegressor")
# f_tree_train,f_tree_test  = run_estimator(k_enum,ensemble.ExtraTreesRegressor(n_estimators=200),train_X, train_y, test_X, test_y=None, test_X2=None)

# test_out = pd.DataFrame()
# test_out['id'] = df_test['id']
# test_out['formation_energy_ev_natom'] = cat_form_test3
# test_out['bandgap_energy_ev'] = xgb_test_band
#
# print(test_out[test_out['bandgap_energy_ev']<0]['bandgap_energy_ev'])
# test_out['formation_energy_ev_natom'] = test_out['formation_energy_ev_natom'].apply(lambda x:abs(x))
# print(test_out[test_out['formation_energy_ev_natom']<0]['formation_energy_ev_natom'])
# test_out.to_csv('sub.csv',index=False)
#
# print("LinRegressor")
# run_estimator(k_enum,linear_model.LinearRegression(),train_X, train_y, test_X, test_y=None, test_X2=None)

#print("ElasticNetCV")
#run_estimator(k_enum,linear_model.MultiTaskElasticNetCV(l1_ratio=0.01, eps=0.0001),train_X, train_y, test_X, test_y=None, test_X2=None)



# print("ExtraTreeRegressor")
# run_estimator(k_enum,ensemble.ExtraTreesRegressor(n_estimators=500),train_X, train_y, test_X, test_y=None, test_X2=None)

sys.exit()
# pd_train = pd.DataFrame(pred_train)
# df_train['lin_test_form'] = pd_train[0]
# df_train['lin_test_band'] = pd_train[1]

# y_pred = pred_full_test/6.0
# y_pred[y_pred <= 0] = 1e-6
# df_test['lin_test_form'] = y_pred[:,0]/6.0
# df_test['lin_test_band'] = y_pred[:,1]/6.0


test_out = pd.DataFrame()
test_out['id'] = df_test['id']
#test_out['formation_energy_ev_natom'] = pred_full_test[:,0]/6.0
test_out['bandgap_energy_ev'] = pred_full_test[:,1]/6.0

# pd_train = pd.DataFrame(pred_train)
# df_train['lin_test_form'] = pd_train[0]
# df_train['lin_test_band'] = pd_train[1]
#
# y_pred = pred_full_test/6.0
# y_pred[y_pred <= 0] = 1e-6
# df_test['lin_test_form'] = y_pred[:,0]/6.0
# df_test['lin_test_band'] = y_pred[:,1]/6.0

cols_to_drop = ['id','formation_energy_ev_natom','bandgap_energy_ev']

train_X = df_train.drop(cols_to_drop, axis=1)
test_X = df_test.drop(['id'], axis=1)


cv_scores = []
pred_full_test = 0
pred_train = np.zeros([df_train.shape[0],2])
params_data = {}
col = ['formation_energy_ev_natom','bandgap_energy_ev']
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'
train_y = df_train[col]

kf = model_selection.KFold(n_splits=6, shuffle=True, random_state=4)
cv_score = []
pred_test_full = 0

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y.loc[dev_index], train_y.loc[val_index]
    pred_val_y, pred_test_y, model1, model2 = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0)
    # pred_val_y, pred_test_y, model = runLinRegres(dev_X, dev_y, val_X, val_y, test_X)
    # pred_val_y, pred_test_y, model = runRandomForest(dev_X, dev_y, val_X, val_y, test_X)
    #pred_val_y, pred_test_y, model = runLinRegres(dev_X, dev_y, val_X, val_y, test_X,polyn=False)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    print(rmsle(val_y, pred_val_y))
    cv_scores.append(metrics.mean_squared_log_error(val_y, pred_val_y))


test_out['formation_energy_ev_natom'] = pred_full_test[:,0]/6.0



# print(cv_scores)
# print(pd_train[0], train_y)
#
# plt.scatter(x = pd_train[0],y = train_y['formation_energy_ev_natom'])
# plt.xlabel('predict_form', fontsize=12)
# plt.ylabel('real_form', fontsize=12)
# plt.show()


# df_test['lin_test_form'] = pred_full_test[:,0]/6.0
# df_test['lin_test_band'] = pred_full_test[:,1]/6.0
# print(df_test['lin_test_form'])



#test_out.to_csv('sub.csv',index=False,float_format='%.4f')
print(test_out)
# plt.scatter(x = pd_train[1],y = train_y['bandgap_energy_ev'])
# plt.xlabel('predict_form', fontsize=12)
# plt.ylabel('real_form', fontsize=12)
# plt.show()
test_out.to_csv('sub.csv',index=False,float_format='%.4f',columns=['id','formation_energy_ev_natom','bandgap_energy_ev'])
sys.exit()
#print(test_out[test_out['bandgap_energy_ev']<0]['bandgap_energy_ev'])
#test_out['formation_energy_ev_natom'] = test_out['formation_energy_ev_natom'].apply(lambda x:abs(x))
#print(test_out[test_out['formation_energy_ev_natom']<0]['formation_energy_ev_natom'])
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