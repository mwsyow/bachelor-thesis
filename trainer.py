import wandb
import os
import torch
import argparse
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from nn_utility import *
from config import Config
from dataloader import TabularDataset
from nn_model.TabNNmodel import TabNNmodel
from utils.encoder import OneHotEncoder, ThermometerEncoder, IntegerEncoder
from sklearn.preprocessing import Normalizer, StandardScaler
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, df, features_dic):
        self.device = None
        self.df = df
        self.original_df = None
        self.features_dic = features_dic
        self.save_model = False
        self.log = False
        self.model_dir_path, self.model_name = None, None
        self.train_idx, self.val_idx, self.test_idx = None, None, None
        self.dim_sizes = None
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

    def _set_model(self, dim_sizes, dropouts, class_ratios):
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
        
        self.model = TabNNmodel(n_ord=n_ord, dim_sizes=dim_sizes, n_nom=n_nom, n_numerical=n_cont, n_label=self.n_label, dropouts=dropouts, class_ratios=class_ratios, is_integer_encoder=is_integer_encoder)
        self.model = self.model.to(self.device)
        self.model.double()
          
    def _set_data_loader(self, df, batch_size, shuffle, num_workers):
        assert 'ordinal' in self.encoders, "ordinal encoders undefined"
        dataset = TabularDataset(df, self.encoders, self.features, self.is_clf)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader
        
    def _set_optimizer(self, lr):
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
        )
    def _set_loss(self):
        if self.n_label == 2: 
            self.loss = nn.BCELoss()
        elif self.n_label > 2: 
            self.loss = nn.CrossEntropyLoss()
        else : self.loss = nn.MSELoss()
                    

    def set_train_idx(self, idx):
        self.train_idx = idx
    
    def set_val_idx(self, idx):
        self.val_idx = idx
        
    def set_model_dir(self, path):
        assert os.path.exists(path), "model dir path doesn't exist"
        self.model_dir_path = path
    
    def set_model_name(self, name):
        self.model_name = name
    
    def set_ord_enc(self, ord_enc):
        self.encoders['ordinal'] = ord_enc(self.features_dic['ordinal'], self.features_dic['step_sizes']) if 'step_sizes' in self.features_dic else ord_enc(self.features_dic['ordinal'])
        
    def set_log(self):
        self.log = True
    
    def set_save_model(self):
        self.save_model = True
    
    def set_device(self, device):
        self.device = device 
    
    def process_continuous_data(self, type):
        assert not self.train_idx is None and not self.val_idx is None, "train or val indices must be specified"
        assert type == 'continuous' or type == 'label', "invalid type"
        if self.original_df is None:
            self.original_df = self.df.copy()
        pre_process = StandardScaler()
        pre_process.fit(self.original_df.iloc[self.train_idx][self.features[type]].to_numpy())
        df_scaled = pre_process.transform(self.original_df[self.features[type]].to_numpy())
        self.df_scaled = df_scaled
        df_scaled = pd.DataFrame(df_scaled, columns = self.features[type])
        for col in self.features[type]:
            self.df[col] = df_scaled[col]
        return pre_process
    
    def update_best_model(self, op, cur_val):
        if op(cur_val, self.best_val):
            update = True
            self.best_val = cur_val
        else : update = False
        if update and self.save_model:
            name = f"{self.model_name}.pth" if not self.log else f"{wandb.run.name}.pth"
            path = os.path.join(self.model_dir_path, name)
            torch.save(self.model.state_dict(), path)
            print(f"model: {name} is saved to path: {path}")
        return update
    
    def probability_to_prediction(self,  y_true, y_prob):
        size = y_true.size()
        if self.n_label == 2:
            return y_true, torch.Tensor([1 if val >=0.5 else 0 for val in y_prob]).reshape(size).to(self.device)
        elif self.n_label > 2:
            y_pred = torch.Tensor([torch.argmax(prob) for prob in y_prob]).to(self.device)
            y = torch.Tensor([torch.argmax(true) for true in y_true]).to(self.device)
            return y, y_pred
        else : return y_true, y_prob
        
    def accuracy(self, y_pred, y_true):
        return torch.sum(y_pred == y_true)/y_true.size(dim=0)
    
    def make_class_ratios(self):
        assert not self.train_idx is None and not self.val_idx is None, "train or val indices must be specified"
        train_df = self.df.iloc[self.train_idx]
        length = len(train_df)
        return [len(train_df[train_df[self.features['label'][0]] == val])/length for val in self.features_dic['label'][self.features['label'][0]]]
    
    def train_epoch(self, train_dl):
        losses = []
        self.model.train()
        for i, data_dic in enumerate(train_dl): 
            x_ord, y = data_dic['ordinal'], data_dic['label']
            x_num = data_dic['continuous'] if 'continuous' in data_dic else None
            x_nom = data_dic['nominal'] if 'nominal' in data_dic else None
            
            x_ord, y = x_ord.to(self.device), y.to(self.device)
            if not x_num is None:
                x_num = x_num.to(self.device)
                x_nom = x_nom.to(self.device)
            
            self.optimizer.zero_grad()
            
            y_pred = self.model(x_ord, x_nom, x_num)
            # y_pred, y = torch.squeeze(y_pred), torch.squeeze(y)
            loss = self.loss(y_pred, y)
            losses.append(loss.item())
            
            loss.backward()
            self.optimizer.step()
        return losses
    
    def val_epoch(self, val_dl):
        losses, accs = [], []
        self.model.eval()
        for i, data_dic in enumerate(val_dl): 
            x_ord, y = data_dic['ordinal'], data_dic['label']
            x_num = data_dic['continuous'] if 'continuous' in data_dic else None
            x_nom = data_dic['nominal'] if 'nominal' in data_dic else None
            
            x_ord, y = x_ord.to(self.device), y.to(self.device)
            if not x_num is None:
                x_num = x_num.to(self.device)
                x_nom = x_nom.to(self.device)
            
            y_pred = self.model(x_ord, x_nom, x_num)
            # y_pred, y = torch.squeeze(y_pred), torch.squeeze(y)
            loss = self.loss(y_pred.detach(), y.detach())
            if self.is_clf:  
                y, y_pred = self.probability_to_prediction(y, y_pred)
                acc = self.accuracy(y_pred, y)
                accs.append(acc.item())
            else:
                mae = nn.L1Loss()
                accs.append(mae(y_pred, y).item())
            losses.append(loss.item())
        return losses, accs 
    
    def train(self, config):
        assert not self.train_idx is None and not self.val_idx is None, "train or val indices must be specified"
        class_ratios = self.make_class_ratios() if self.is_clf else None
        self._set_model(config["dim_sizes"], config["dropouts"], class_ratios)
        self.dim_sizes = config["dim_sizes"]
        self._set_optimizer(config["lr"])
        self._set_loss()
        train_dl = self._set_data_loader(self.df.iloc[self.train_idx], config["batch_size"], config["shuffle"], config["num_workers"])
        val_dl = self._set_data_loader(self.df.iloc[self.val_idx], config["batch_size"], config["shuffle"], config["num_workers"])
        stop_criterion = config['early_stopping']
        margin = config['margin']
        patience_cur = 0
        self.best_val, op = np.inf, lambda x, y: x < y - margin
        if self.log:
            wandb.watch(self.model, log='all', log_freq=config['batch_size']/10 if config['batch_size']/10 >= 1 else 1)
        for epoch in range(1, config['epoch']+1):
            train_losses = self.train_epoch(train_dl)
            val_losses, accs = self.val_epoch(val_dl)
            wandb_status = {
                "epoch": epoch,
                "train_loss": sum(train_losses)/len(train_losses),
                "val_loss": sum(val_losses)/len(val_losses),
                }
            if self.is_clf:
                wandb_status.update({"accuracy": sum(accs)/len(accs)})
                print("[epoch: %3d/%3d] train loss: %3f, test loss: %3f, accuracy: %3f" % (epoch, config['epoch'], wandb_status["train_loss"], wandb_status["val_loss"], wandb_status['accuracy']))
            else : 
                wandb_status.update({"mae": sum(accs)/len(accs)})
                print("[epoch: %3d/%3d] train loss: %3f, test loss: %3f, mae: %3f" % (epoch, config['epoch'], wandb_status["train_loss"], wandb_status["val_loss"], wandb_status["mae"]))
            if self.log:
                wandb.log(wandb_status)
            if self.update_best_model(op, wandb_status["val_loss"]):
                patience_cur = 0
            else: patience_cur += 1
  
            if patience_cur >= stop_criterion:
                print("Training stopped by early stopping")
                break

if __name__ == "__main__":
    parser =  argparse.ArgumentParser(description="training data")
    parser.add_argument('--mode', type=str, choices=['log', 'sweep'], default='log', help='Config type. Choices are log or sweep')
    args = parser.parse_args()
    config = Config(f"configs/{args.mode}_config.json").config
    
    if config['wandb']['mode'] == 'log' or config['wandb']['mode'] == 'sweep':
        wandb.login(key=config['wandb']['api_key'], relogin=True)
    
    model_name = config['model']['name']
    model_path = config['model']['path']
    for en in ['ohe','te', 'ie']:
        config['model']['encoder'] = en
        for dr in [i/10 for i in range(10)]:
            dr_str = ''.join(str(dr).split('.'))
            config['model']['name'] = model_name+f"_dr{dr_str}"
            model_cur_path = os.path.join(model_path, en, f"dr{dr_str}")
            if not os.path.exists(model_cur_path):
                os.mkdir(model_cur_path)
            config['model']['path'] = model_cur_path
            config['train']['parameters']['dropouts'] = [0.0]+[dr]*len(config['train']['parameters']['dim_sizes'])
            k_fold_validation(k=10, config=config)