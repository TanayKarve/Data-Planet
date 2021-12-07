import pandas as pd
from dataplanet.dataplanet import dataplanet
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('iris.csv')


X = df.drop(columns=['class'])
y = df['class']
X_train,X_test, y_train,y_test = train_test_split(X,y)
    

n_estimators_grid = [25,50,75]
param_list = ['n_estimators_grid']
metric_list = ['accuracy']

EXP_NAME = 'rf iris'

dp = dataplanet(EXP_NAME,param_list,['accuracy'])
dp.set_param_list('n_estimators_grid')

for ne in n_estimators_grid:
        with dp.mlflow.start_run():
            dp.log_params('n_estimators_grid')
            clf = RandomForestClassifier(n_estimators=ne)
            dp.set_model(clf)
            clf.fit(X_train, y_train.ravel())
            y_pred = clf.predict(X_test)
            dp.log(y_test,y_pred)

            dp.log_model(clf)

dp.commit()
models = dp.get_models()

max_acc_URI=max(models)
URI=max_acc_URI[0]+'/model'
loaded_model = dp.mlflow.pyfunc.load_model(URI)

y_pred = loaded_model.predict(X_test)
