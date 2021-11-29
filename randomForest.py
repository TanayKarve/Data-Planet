import csv
import pandas as pd
from collections import Counter
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from dataplanet import dataplanet
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import metrics
import pickle
import math
import re
#import enchant
import os
import glob
import numpy as np
np.random.seed(512)

#dataverse
xtrain = pd.read_csv('data_train.csv')
xtest = pd.read_csv('data_test.csv')


xtrain = xtrain.sample(frac=1,random_state=100).reset_index(drop=True)
print(len(xtrain))

y_train = xtrain.loc[:,['y_act']]
y_test = xtest.loc[:,['y_act']]

dict_label = {
    'numeric': 0,
    'categorical': 1,
    'datetime': 2,
    'sentence': 3,
    'url': 4,
    'embedded-number': 5,
    'list': 6,
    'not-generalizable': 7,
    'context-specific': 8
}

y_train['y_act'] = [dict_label[i] for i in y_train['y_act']]
y_test['y_act'] = [dict_label[i] for i in y_test['y_act']]
y_train

useStats = 1
useAttributeName = 1
useSample1 = 0
useSample2 = 0
## Using descriptive stats and attribute name

def dataverse(x):
  """
  dataverse processing
  """
  return x


@dataverse
def ProcessStats(data,y):

    data1 = data[['total_vals', 'num_nans', '%_nans', 'num_of_dist_val', '%_dist_val', 'mean', 'std_dev', 'min_val', 'max_val','has_delimiters', 'has_url', 'has_email', 'has_date', 'mean_word_count',
       'std_dev_word_count', 'mean_stopword_total', 'stdev_stopword_total',
       'mean_char_count', 'stdev_char_count', 'mean_whitespace_count',
       'stdev_whitespace_count', 'mean_delim_count', 'stdev_delim_count',
       'is_list', 'is_long_sentence']]
    data1 = data1.reset_index(drop=True)
    data1 = data1.fillna(0)

    y.y_act = y.y_act.astype(float)

    return data1


vectorizerName = CountVectorizer(ngram_range=(2, 2), analyzer='char')
vectorizerSample = CountVectorizer(ngram_range=(2, 2), analyzer='char')

def FeatureExtraction(data,data1,flag):

    arr = data['Attribute_name'].values
    arr = [str(x) for x in arr]

    arr1 = data['sample_1'].values
    arr1 = [str(x) for x in arr1]
    arr2 = data['sample_2'].values
    arr2 = [str(x) for x in arr2]
    arr3 = data['sample_3'].values
    arr3 = [str(x) for x in arr3]
    print(len(arr1),len(arr2))
    if flag:
        X = vectorizerName.fit_transform(arr)
        X1 = vectorizerSample.fit_transform(arr1)
        X2 = vectorizerSample.transform(arr2)

    else:
        X = vectorizerName.transform(arr)
        X1 = vectorizerSample.transform(arr1)
        X2 = vectorizerSample.transform(arr2)

#     print(f"> Length of vectorized feature_names: {len(vectorizer.get_feature_names())}")

    attr_df = pd.DataFrame(X.toarray())
    sample1_df = pd.DataFrame(X1.toarray())
    sample2_df = pd.DataFrame(X2.toarray())
    print(len(data1),len(attr_df),len(sample1_df),len(sample2_df))

    if useSample1: data2 = sample1_df
    if useSample2: data2 = sample2_df

    data2 = pd.concat([data1, attr_df], axis=1, sort=False)
    print(len(data2))
    for column in data2.columns:
      if data2[column].dtype == int:
          data2[column] = data2[column].astype(float)

    return data2

xtrain1 = ProcessStats(xtrain,y_train)
xtest1 = ProcessStats(xtest,y_test)


X_train = FeatureExtraction(xtrain,xtrain1,1)
X_test = FeatureExtraction(xtest,xtest1,0)


X_train_new = X_train.reset_index(drop=True)
y_train_new = y_train.reset_index(drop=True)
X_train_new = X_train_new.values
y_train_new = y_train_new.values
X_train_new



k = 2
kf = KFold(n_splits=k,random_state = 100, shuffle=True)
avg_train_acc,avg_test_acc = 0,0

# n_estimators_grid = [5,25,50,75,100,500]
# max_depth_grid = [5,10,25,50,100,250]

n_estimators_grid = [25,50,75]
max_depth_grid = [25,50,75]

#avgsc_lst,avgsc_train_lst,avgsc_hld_lst = [],[],[]
#avgsc,avgsc_train,avgsc_hld = 0,0,0

best_param_count = {'n_estimator': {}, 'max_depth': {}}
# dataplanet.set_param_list('max_depth','n_estimator')

i=0

EXP_NAME='rf_iris'
param_list = ['max_depth','n_estimator']
metric_list = ['accuracy']
dp = dataplanet.dataplanet(EXP_NAME, param_list, metric_list)

print(dp.experiment_name)

for train_index, test_index in kf.split(X_train_new):
#     if i==1: break
    i=i+1
    X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
    y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]
    X_train_train, X_val,y_train_train,y_val = train_test_split(X_train_cur,y_train_cur, test_size=0.25,random_state=100)

    print('='*10)
    for ne in n_estimators_grid:
        for md in max_depth_grid:
          with dp.start_run():
              #Log parameters
              dp.log_params(md, ne)

              clf = RandomForestClassifier(n_estimators=ne,max_depth=md)
              dp.set_model(clf)
              clf.fit(X_train_train, y_train_train.ravel())
              sc = clf.score(X_val, y_val)
              # print(f"[n_estimator: {ne}, max_depth: {md}, accuracy: {sc}]")
              signature = dp.get_model_signature(X_train, clf.predict(X_train))
              dp.log(y_val, clf.predict(X_val))
              dp.log_model(clf)

models = dp.get_models()

max_acc_URI = max(models,key=lambda x: x[1])
URI = max_acc_URI[0]+'/clf'
loaded_model = pickle.load(open(URI+'/model.pkl','rb'))
y_pred = loaded_model.predict(X_val)
# dataplanet.log(y_val, y_pred)
#print(accuracy_score(y_val,y_pred))

#reduced LoC by ~40%
_="""
            if bestscore < sc:
                bestne = ne
                bestmd = md
                bestscore = sc
                bestPerformingModel = clf
    if str(bestne) in best_param_count['n_estimator']:
        best_param_count['n_estimator'][str(bestne)] += 1
    else:
        best_param_count['n_estimator'][str(bestne)] = 1
    if str(bestmd) in best_param_count['max_depth']:
        best_param_count['max_depth'][str(bestmd)] += 1
    else:
        best_param_count['max_depth'][str(bestmd)] = 1
    bscr_train = bestPerformingModel.score(X_train_cur, y_train_cur)
    bscr = bestPerformingModel.score(X_test_cur, y_test_cur)
    bscr_hld = bestPerformingModel.score(X_test, y_test)
    avgsc_train_lst.append(bscr_train)
    avgsc_lst.append(bscr)
    avgsc_hld_lst.append(bscr_hld)
    avgsc_train = avgsc_train + bscr_train
    avgsc = avgsc + bscr
    avgsc_hld = avgsc_hld + bscr_hld
    print()
    print(f"> Best n_estimator: {bestne} || Best max_depth: {bestmd}")
    print(f"> Best training score: {bscr_train}")
    print(f"> Best test score: {bscr}")
    print(f"> Best held score: {bscr_hld}")
print('='*10)
print(avgsc_train_lst)
print(avgsc_lst)
print(avgsc_hld_lst)
print(avgsc_train/k)
print(avgsc/k)
print(avgsc_hld/k)
y_pred = bestPerformingModel.predict(X_test)
bscr_hld = bestPerformingModel.score(X_test, y_test)
print(bscr_hld)
"""
X_train_ = X_train
