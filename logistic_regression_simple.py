import pandas as pd
from dataplanet.dataplanet import dataplanet
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('iris.csv')

X = df.drop(columns=['class'])
y = df['class']
X_train,X_test, y_train,y_test = train_test_split(X,y)
    

reg_space = [0.01, 0.1, 1]
param_list = ['reg_space']
metric_list = ['accuracy']

EXP_NAME = 'iris log regression'


dp = dataplanet(EXP_NAME,param_list,metric_list)
dp.set_param_list(reg_space)
for reg in reg_space:
        with dp.mlflow.start_run():
            dp.log_params('reg_space')

            clf = LogisticRegression(penalty='l2', multi_class='multinomial', solver='lbfgs', C=reg)
            dp.set_model(clf)

            clf.fit(X_train, y_train)
            dp.log(y_test,clf.predict(X_test))
            dp.log_model(clf)

dp.commit()
models = dp.get_models()

max_acc_URI=max(models)
URI=max_acc_URI[0]+'/model'
loaded_model = dp.mlflow.pyfunc.load_model(URI)

y_pred = loaded_model.predict(X_test)
