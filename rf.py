import pandas as pd
# import sortinghat.pylib as sh
from dataplanet.dataplanet import dataplanet
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('iris.data')

#dataDownstream = pd.read_csv('SortingHatLib/data/adult.csv')
# dataFeaturized = sh.FeaturizeFile(df)
# data=sh.FeatureExtraction(dataFeaturized)
X = df.drop(columns=['class'])
y = df['class']
X_train,X_test, y_train,y_test = train_test_split(X,y)

k = 2
kf = KFold(n_splits=k)



n_estimators_grid = [25,50,75]
max_depth_grid = [25,50,75]
EXP_NAME = 'iris'

dataPlanet = dataplanet(EXP_NAME,['max_depth','n_estimator'],['accuracy'],)
for ne in n_estimators_grid:
    for md in max_depth_grid:
        with dataPlanet.mlflow.start_run():
            #dataPlanet.log_params(dataPlanet.get_param_list())
            clf = RandomForestClassifier(n_estimators=ne,max_depth=md)
            dataPlanet.set_model(clf)
            clf.fit(X_train, y_train.ravel())
            y_pred = clf.predict(X_test)
            #dataPlanet.log(y_test,y_pred)
            dataPlanet.log_model(clf)