import mlflow
from mlflow.models import signature
from mlflow.entities import experiment
from collections import defaultdict
import importlib
import sklearn
import sklearn.metrics
from urllib.parse import unquote, urlparse
import os
import json
import pickle
import subprocess

class dataplanet:

    def __init__(self, experiment_name, param_list, metric_list,UI_PORT=8000):
        self.dataverse_url = "http://dataverse-dev.localhost:8085"
        self.experiment_name = experiment_name
        self.param_list = param_list
        self.metric_list = metric_list
        self.mlflow = mlflow
        self.mlflow_url = f"http://localhost:{UI_PORT}"
        subprocess.Popen(['mlflow',"ui", "--port", str(UI_PORT)])

    def set_tracking_uri(self, tracking_uri):
        self.tracking_uri = tracking_uri
        self.mlflow.set_tracking_uri(self.tracking_uri)

    def set_dataverse_url(self, dataverse_url):
        self.dataverse_url = dataverse_url

    def get_dataverse_url(self):
        return self.dataverse_url

    def set_param_list(self, *param_list):
        self.param_list = param_list

    def get_param_list(self):
        return self.param_list

    def set_metric_list(self, *metric_list):
        self.metric_list = metric_list

    def get_metric_list(self):
        return self.metric_list

    def set_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name
        self.mlflow.set_experiment(self.experiment_name)

    def get_experiment_name(self):
        return self.experiment_name

    def set_model(self,model):
        self.model = model
        self.set_model_library()

    def get_model(self):
        return self.model

    def get_models(self):
        models=[]
        # for ri in self.mlflow.list_run_infos(self.mlflow.get_experiment_by_name(self.experiment_name)):
        for ri in self.mlflow.list_run_infos('0'):
            run = self.mlflow.get_run(ri.run_id)
            for metric in self.metric_list:
                try:
                    artifact_uri = run.info.artifact_uri
                    m = run.data.metrics[metric]
                    models.append((artifact_uri,metric))
                except KeyError:
                    pass
        if len(models) == 0:
            models = [(artifact_uri,1)]
            
        return models
    
    def get_model_summary(self,lib):
        model_name = self.model.__class__.__name__

        if lib in {'keras','tensorflow'}:
            params = self.get_config()
        elif lib == 'pytorch':
            params = dict(self.model.named_children())
        elif lib == 'sklearn':
            params = self.model.get_params()
        meta = {'model type':model_name,'parameters':params}

        return meta 

    def get_mlflow_ui(self):
        return self.mlflow.get_tracking_uri()

    
    def get_model_meta(self):
        model_type = self.get_model_library()
        model_support = {'keras','torch','tensorflow','sklearn'}
        if model_type not in model_support:
            raise NotImplementedError(f'Library {model_type} not implemented')
        else:
            model_meta = self.get_model_summary(model_type)
        return model_meta

    def commit(self,by_metric='accuracy'):
        model_meta=self.get_model_meta()
        models=[]
        for ri in self.mlflow.list_run_infos('0'):
            run = self.mlflow.get_run(ri.run_id)
            try:
                artifact_uri = run.info.artifact_uri
                metric = run.data.metrics[by_metric]
                models.append((artifact_uri,metric))
            except KeyError:
                if len(models) ==0:
                    models.append((artifact_uri,1))
                pass
        try:
            max_acc_URI=max(models,key=lambda x: x[1])
        except ValueError:
            max_acc_URI = models[0]
        URI = unquote(urlparse(max_acc_URI[0]).path)
        model_object = pickle.load(open(URI+'/model/model.pkl','rb'))
        payload ={'URI':URI,'model':model_meta,'mlflow_url':self.mlflow_url}
        json.dump(payload,open('model_metadata.json','w'))

    # def log_params(self, **param_values):
    #     for param in self.param_list:
    #         mlflow.log_param(param, param_values[param])

    def log_params(self, *param_values):
        values = iter(param_values)
        for param in self.param_list:
            self.mlflow.log_param(param, next(values))

    def start_run(self):
        return self.mlflow.start_run()

    def get_model_signature(self, features, predictions):
        self.model_signature = signature.infer_signature(features, predictions)
        return self.model_signature

    def set_model_signature(self, model_signature):
        self.model_signature = model_signature

    def log(self, labels, predictions):
        self.predictions = predictions
        self.labels = labels
        self.log_metrics()

    def set_param_count(self):
        self.param_count = len(self.param_list)

    def get_param_count(self):
        return self.param_count

    def set_model_library(self):
        # self.model_library = str(type(self.model)).split('.')[0].split('\'')
        self.model_library = self.model.__class__.__bases__[0].__module__.split('.')[0]

    def get_model_library(self):
        return self.model_library

    def log_metrics(self):
        if self.model_library == 'sklearn':
            self.log_sklearn_metrics()

    def log_model(self,model):
        model_library = self.get_model_library()
        if model_library == 'sklearn':
            self.mlflow.sklearn.log_model(model,'model')
            
    def log_sklearn_metrics(self):
        sklearn_metrics = {'accuracy':'accuracy_score',
            'adjusted mis':'adjusted_mutual_info_score',
            'adjusted rand':'adjusted_rand_score',
            'auc':'auc',
            'average precision':'average_precision_score',
            'balanced accuracy':'balanced_accuracy_score',
            'brier score loss': 'brier_score_loss',
            'calinski harabasz':'calinski_harabasz_score',
            'check scoring':'check_scoring',
            'classification report':'classification_report',
            'cluster':'cluster',
            'cohen kappa':'cohen_kappa_score',
            'completeness':'completeness_score',
            'confusion matrix':'confusion_matrix',
            'consensus':'consensus_score',
            'coverage error':'coverage_error',
            'd2 tweedie':'d2_tweedie_score',
            'davies bouldin':'davies_bouldin_score',
            'dcg':'dcg_score',
            'det':'det_curve',
            'euclidean distances':'euclidean_distances',
            'explained variance':'explained_variance_score',
            'f1':'f1_score',
            'fbeta':'fbeta_score',
            'fowlkes mallows':'fowlkes_mallows_score',
            'get scorer':'get_scorer',
            'hamming loss':'hamming_loss',
            'hinge loss':'hinge_loss',
            'homogeneity completeness v measure':'homogeneity_completeness_v_measure',
            'homogeneity':'homogeneity_score',
            'jaccard':'jaccard_score',
            'label ranking average precision':'label_ranking_average_precision_score',
            'label ranking loss':'label_ranking_loss',
            'log loss':'log_loss',
            'make scorer':'make_scorer',
            'matthews corrcoef':'matthews_corrcoef',
            'max error':'max_error',
            'mae':'mean_absolute_error',
            'percent mae':'mean_absolute_percentage_error',
            'mgd':'mean_gamma_deviance',
            'mpl':'mean_pinball_loss',
            'mpd':'mean_poisson_deviance',
            'mse':'mean_squared_error',
            'log mse':'mean_squared_log_error',
            'mtd':'mean_tweedie_deviance',
            'median absolute error':'median_absolute_error',
            'multilabel confusion matrix':'multilabel_confusion_matrix',
            'mis':'mutual_info_score',
            'nan euclidean distances':'nan_euclidean_distances',
            'ndcg':'ndcg_score',
            'normalized mis':'normalized_mutual_info_score',
            'pair confusion matrix':'pair_confusion_matrix',
            'pairwise':'pairwise',
            'pairwise distances':'pairwise_distances',
            'pairwise distances argmin':'pairwise_distances_argmin',
            'pairwise distances argmin min':'pairwise_distances_argmin_min',
            'pairwise distances chunked':'pairwise_distances_chunked',
            'pairwise_kernels':'pairwise_kernels',
            'plot confusion matrix':'plot_confusion_matrix',
            'plot det':'plot_det_curve',
            'plot precision recall':'plot_precision_recall_curve',
            'plot roc': 'plot_roc_curve',
            'precision recall':'precision_recall_curve',
            'precision recall fscore support':'precision_recall_fscore_support',
            'precision':'precision_score',
            'r2':'r2_score',
            'rand':'rand_score',
            'recall':'recall_score',
            'roc accuracy':'roc_auc_score',
            'roc':'roc_curve',
            'silhouette samples':'silhouette_samples',
            'silhouette':'silhouette_score',
            'top k accuracy':'top_k_accuracy_score',
            'v meansure':'v_measure_score',
            'zero one loss':'zero_one_loss'}

        for metric in self.metric_list:
            if metric in sklearn_metrics:
                metric_val = getattr(sklearn.metrics, sklearn_metrics[metric])(self.labels, self.predictions)
                self.mlflow.log_metric(metric, metric_val)
            else:
                raise ValueError('Metric Not Found')
