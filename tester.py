import torch
import os
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import utility
from dataloader import TabularDataset
from nn_model.TabNNmodel import TabNNmodel
from utils.encoder import OneHotEncoder, IntegerEncoder
from sklearn.metrics import classification_report


class Tester:
    def __init__(self, df, features_dic):
        self.device = None
        self.df = df
        self.original_df = None
        self.features_dic = features_dic
        self.model_path = None
        self._set_features_and_encoders()

    def _set_features_and_encoders(self):
        assert 'label' in self.features_dic and 'ordinal' in self.features_dic
        self.encoders = {}
        self.features = {}
        feat_list = []
        if 'nominal' in self.features_dic:
            self.encoders['nominal'] = OneHotEncoder(self.features_dic['nominal'])
            self.features['nominal'] = list(self.features_dic['nominal'].keys())
            feat_list += self.features['nominal']
        if isinstance(self.features_dic['label'], dict):
            if len(list(self.features_dic['label'].values())[0]) > 2: 
                self.encoders['label'] = OneHotEncoder(self.features_dic['label'])
            else:  self.encoders['label'] = IntegerEncoder(self.features_dic['label'], is_label=True)
            self.features['label'] = list(self.features_dic['label'].keys())
            self.is_clf = True
        else: 
            self.features['label'] = self.features_dic['label']
            self.is_clf = False
        self.features['ordinal'] = list(self.features_dic['ordinal'].keys())
        feat_list += self.features['label'] + self.features['ordinal']
        cont_feat = list(set(self.df.columns) - set(feat_list))
        if len(cont_feat) > 0:
            self.features['continuous'] = cont_feat
    
    def _set_model(self, dim_sizes):
        assert not self.model_path is None, "model path must be specified"
        assert 'ordinal' in self.encoders, 'encoder for ordinal features is undefined'
        if self.encoders['ordinal'].__str__() == "IntegerEncoder":
            n_ord = len(self.features['ordinal'])
            is_integer_encoder = True
        else: 
            n_ord = sum([len(val) for val in self.features_dic['ordinal'].values()])
            is_integer_encoder = False
        
        if 'nominal' in self.encoders:
            if self.encoders['nominal'].__str__() == "IntegerEncoder":
                n_nom = len(self.features['nominal'])
            else: n_nom = sum([len(val) for val in self.features_dic['nominal'].values()])
        else: n_nom = 0
        
        self.n_label = sum([len(val) for val in self.features_dic['label'].values()]) if self.is_clf else 0
        n_cont = len(self.features_dic['continuous']) if 'continuous' in self.features_dic else 0
        
        self.model = TabNNmodel(n_ord=n_ord, dim_sizes=dim_sizes, n_nom=n_nom, n_numerical=n_cont, n_label=self.n_label, is_integer_encoder=is_integer_encoder)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.double()
        

    def _set_data_loader(self, df, batch_size, shuffle, num_workers):
        assert 'ordinal' in self.encoders, "ordinal encoders undefined"
        dataset = TabularDataset(df, self.encoders, self.features, self.is_clf)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader
            
    def _set_eval_metric(self):
        if self.is_clf:
            self.eval_metric = {'clf_rep': lambda y_true, y_pred: classification_report(y_true, y_pred, output_dict=True)}
        else : self.eval_metric = {
                            'RMSE': lambda x,y : mean_squared_error(x, y, squared=False), 
                            'MAE': mean_absolute_error
                            }
    def set_ord_enc(self, ord_enc):
        self.encoders['ordinal'] = ord_enc(self.features_dic['ordinal'], self.features_dic['step_sizes']) if 'step_sizes' in self.features_dic else ord_enc(self.features_dic['ordinal'])
    
    def set_model_path(self, path):
        assert os.path.exists(path), "path does not exist"
        self.model_path = path
            
    def set_device(self, device):
        self.device = device
            
    def process_continuous_data(self, pre_process, type):
        assert type == 'continuous' or type == 'label', "invalid type"
        if self.original_df is None:
            self.original_df = self.df.copy()
        df_scaled = pre_process.transform(self.original_df[self.features[type]].to_numpy())
        df_scaled = pd.DataFrame(df_scaled, columns = self.features[type])
        for col in self.features[type]:
            self.df[col] = df_scaled[col]
            
    def probability_to_prediction(self, y_true, y_prob):
        shape = y_true.shape
        if self.n_label == 2:
            return y_true, np.array([1 if val >=0.5 else 0 for val in y_prob]).reshape(shape)
        elif self.n_label > 2:
            y_pred = np.array([np.argmax(prob) for prob in y_prob])
            y = np.array([np.argmax(true) for true in y_true])
            return y, y_pred
        else : return y_true, y_prob
            
    def evaluate(self, y_true, y_pred):
        loss_dic = {}
        if self.device.type == 'cuda':
            y_true, y_pred = y_true.to(torch.device('cpu')), y_pred.to(torch.device('cpu'))
        y_true, y_pred = y_true.detach().numpy(), y_pred.detach().numpy()  
        if self.is_clf:  
            y_true, y_pred = self.probability_to_prediction(y_true, y_pred)
        for k, metric in self.eval_metric.items():
            loss_dic[k] = metric(y_true, y_pred)
        return loss_dic

    def test(self, config):
        self._set_model(config["dim_sizes"])
        self._set_eval_metric()
        if config['batch_size'] == 'all':
            batch_size = len(self.df)
        else: batch_size = config['batch_size']
        test_dl = self._set_data_loader(self.df, batch_size, config["shuffle"], config["num_workers"])
        self.model.eval()
        result = {k: [] for k in self.eval_metric.keys()}
        for data_dic in test_dl: 
            x_ord, y = data_dic['ordinal'], data_dic['label']
            x_num = data_dic['continuous'] if 'continuous' in data_dic else None
            x_nom = data_dic['nominal'] if 'nominal' in data_dic else None
            
            x_ord, y = x_ord.to(self.device), y.to(self.device)
            if not x_num is None:
                x_num = x_num.to(self.device)
                x_nom = x_nom.to(self.device)
            
            y_pred = self.model(x_ord, x_nom, x_num)
            
            batch_eval = self.evaluate(y, y_pred)
            for k, v in batch_eval.items():
                result[k].append(v)
        if self.is_clf:
            for k in self.eval_metric.keys():
                result[k] = utility.concat_dic(*result[k])
                result[k] = utility.mean_std_dic(result[k], only_mean=True)
        else:
            result = {k: sum(v)/len(v) for k, v in result.items()}        
        return result
            