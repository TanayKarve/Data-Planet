# Data-Planet

Domain Scientists perform a range on experiments on different ML procedures. Currently, the workflow involves manually performing operations like data ETL, model training, selection, validation. Additionally management of datasets and model configuration itself is a huge overhead. We create a unified platform that encapsulates these procedures and automates this pipeline, by integrating two robust frameworks MLFlow and Dataverse into a platform called dataplanet. MLFlow provides a streamlined way to automate the machine learning workflow and makes model versioning, packing and tracking feasible. Whereas, Dataverse is a dataset management tool where datasets and their associated metadatas can be stored along with inbuilt version control. Dataplanet makes experiment and data management effortless and also provides one-click reproducibility, thus making ML even more accessible.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AGDNz7usLsmkVsLYNzUX_i7e4KtA6Qll)

## Usage

### Step 1: Clone and Import

We provide a python library which is a wrapper around the MLFlow library and has Dataverse integration as well. It is open source and is hosted on Github.

```bash
git clone https://github.com/TanayKarve/Data-Planet

cd Data-Planet
pip install mlflow
```

```python
import pandas as pd
from dataplanet.dataplanet import dataplanet
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
```
### Step 2: Data preprocessing

Every ML Experiment workflow has an associated data preprocessing pipeline. For the sake of this tutorial, we'll keep it simple.

```python
df = pd.read_csv('data/ig.csv')
y = df['FOLLOWING']
df=pd.concat([df,pd.get_dummies(df['CATEGORIES 1'].reset_index(drop=True))],axis=1)
df=pd.concat([df,pd.get_dummies(df['CATEGORIES 2'].reset_index(drop=True))],axis=1)
X = df.drop(['FOLLOWING','BRAND','CATEGORIES 1','CATEGORIES 2'],axis=1)
X_train,X_test, y_train,y_test = train_test_split(X,y)

#
#your workflow code here...
#
```

### Step 3: Hyperparameter grid
The core feature of dataplanet is ML experiment reproducibility. Thus, the library supports hyperparameter search logging, thereby keeping track of all model configurations and their corresponding results.

The below code snippet creates a new dataplanet object that initializes a new experiment named EXP_NAME to run your models with the the list of parameters (param_list) to be logged and the metrics (metric_list) that has to be calculated. dp.set_param_list(reg_space) Once you set your parameters and metrics, you can run your model to generate logs for different configuration of their values. The code sets the parameter space (reg_space) for which the model will run.

```python
n_estimators_grid = [25,50,75]
param_list = ['n_estimators_grid']
metric_list = ['accuracy']
EXP_NAME = 'rf iris'
dp = dataplanet(EXP_NAME,param_list,['accuracy'])
dp.set_param_list('n_estimators_grid')
```

### Step 4: Training
The code below runs a logistic regression model for the list of values provided the in the parameter space (reg_space)

and logs these values for each model run using log_params() func- tion. Once you run your model, you can then log your results using

which dataplanet will calculate the metrics you specified into your experiment. DataPlanet lets you extract the details of your models and their metadata for reproducibility using the above function. This gener- ates a json file which includes prerequisite packages, model config- urations, model signature and artifacts that is populated into your Dataverse along with your dataset. Alternatively, it also generates artifacts for each model that you run under an experiment so you can chose which model to run depending on your requirements.

```python
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
```

### Step 5: Reproducibility

Additonally, instead of selecting a model manually you can select the best performing model by default. The get_models() function in the above code snippet does exactly that. You can then load your model and run it as shown above. The full list of methods available in the DataPlanet API is available at the appendix. This json file containing the metadata of the project is uploaded onto Dataverse UI. After the user publishes the project on DataPlanet, metadata fields from the json file are automatically popu- lated. The fields can be searched using the text search functionality offered by Dataverse’s UI. Figure 6 shows the search functionality.


```python
models = dp.get_models()

max_acc_URI=max(models)
URI=max_acc_URI[0]+'/model'
loaded_model = dp.mlflow.pyfunc.load_model(URI)

y_pred = loaded_model.predict(X_test)
```
