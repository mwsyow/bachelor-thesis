import os
import pickle
import joblib
import glob
import torch
import wandb
import pandas as pd
from trainer import Trainer
from tester import Tester
from sklearn.model_selection import train_test_split
from utils import utility
from utils.encoder import OneHotEncoder, ThermometerEncoder, IntegerEncoder
from sklearn.model_selection import StratifiedKFold, KFold


def open_dataset(path, data_name, dic_name):
    data_path = os.path.join(path, data_name+".csv")
    dic_path = os.path.join(path, dic_name+"_dic.pkl")
    df = pd.read_csv(data_path)
    with open(dic_path, 'rb') as f:
        feature_dic = pickle.load(f)
    return df, feature_dic

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_encoders(encoder_name):
    encoder_dic = {
        'o': OneHotEncoder, 'ohe': OneHotEncoder, 'onehotencoder': OneHotEncoder,
        't': ThermometerEncoder, 'te': ThermometerEncoder, 'thermometerencoder': ThermometerEncoder,
        'i': IntegerEncoder, 'ie': IntegerEncoder, 'integerencoder': IntegerEncoder,
    }
    encoders = []
    if isinstance(encoder_name, str):
        if encoder_name == 'all':
            return [OneHotEncoder, ThermometerEncoder, IntegerEncoder]
        else : 
            encoders.append(encoder_dic[encoder_name.lower()])
            return encoders
    else:    
        for name in encoder_name:
            encoders.append(encoder_dic[name.lower()])
        return encoders
    
def sweep_train(trainer, config=None):
            with wandb.init(config=config):
                config = wandb.config
                trainer.train(config)
                

def train_log(trainer, mode, config, entity=None, project=None, group=None, name=None, count=None):
    if mode == "sweep":
        trainer.set_log()
        sweep_id = wandb.sweep(config, entity=entity, project=project)
        wandb.agent(sweep_id, function= lambda cfg=None: sweep_train(trainer=trainer, config=cfg), count=count)
        os.system(f"wandb sweep --stop {sweep_id}")
    elif mode == "log": 
        trainer.set_log()
        with wandb.init(entity=entity, project=project, group=group, name=name, config=config):
            config = wandb.config
            trainer.train(config['parameters'])
            wandb.finish()
    else: 
        trainer.train(config['parameters'])
    
def k_fold_validation(k, config, return_test_set=False, return_pre_process=False):
    data_path = os.path.join(config["dataset"]["path"], config["dataset"]["name"])
    df, features_dic = open_dataset(data_path, config["dataset"]["name"], config["dataset"]["name"])
    
    
    
    if isinstance(features_dic['label'], dict):
        train_df, test_df = train_test_split(df, test_size=config['dataset']['split'], random_state=config['dataset']['seed'], stratify=df[features_dic['label'].keys()])
        kf = StratifiedKFold(k) 
        split = kf.split(train_df,y=train_df[features_dic['label'].keys()])
    else: 
        train_df, test_df = train_test_split(df, test_size=config['dataset']['split'], random_state=config['dataset']['seed'])
        kf = KFold(k)
        split = kf.split(train_df)

    if config['dataset']['save_test']:
        test_path = os.path.join(data_path, config["dataset"]["name"] + "_test.csv")
        test_df.reset_index(drop=True).to_csv(test_path, index=False)
    
    mode = config['wandb']['mode']
    encoders = get_encoders(config['model']['encoder'])
    project = config['wandb']['project'] if 'project' in config['wandb'] else None
    count = config['wandb']['count'] if 'count' in config['wandb'] else None
    entity = config['wandb']['entity'] if 'entity' in config['wandb'] else None
    group = config['wandb']['group'] if 'group' in config['wandb'] else None
 
    trainer = Trainer(train_df.reset_index(drop=True), features_dic)
    
    device = get_device()
    trainer.set_device(device)
    
    save_path = config["model"]["path"]
    model_name = config["model"]["name"]
    save_model =  config["model"]['save']
    if save_model:
        trainer.set_save_model()
    trainer.set_model_dir(save_path)
    

    pre_processes = {'continuous': [], 'label': []}
    for i, (train_idx, val_idx) in enumerate(split):
        trainer.set_train_idx(train_idx), trainer.set_val_idx(val_idx)
        # if 'continuous' in features_dic:
        #     sc_cont = trainer.process_continuous_data('continuous')
        #     pre_processes['continuous'].append(sc_cont)
        if not trainer.is_clf:
            sc_label = trainer.process_continuous_data('label')
            pre_processes['label'].append(sc_label)
        for encoder in encoders:
            name = f"{model_name}_{encoder.__str__().lower()}_{i}"
            trainer.set_model_name(name)
            trainer.set_ord_enc(encoder), 
            train_log(trainer, mode=mode, config=config['train'], project=project, group=group, name=name, count=count, entity=entity)
    
    if config['dataset']['save_pre_process']:
        for key in pre_processes.keys():
            if len(pre_processes[key])>0:
                for i, sc in enumerate(pre_processes[key]):
                    sc_path = os.path.join(config['dataset']['pre_process_path'], config['dataset']['name'], f'pre_process_{key}_{i}.gz')
                    joblib.dump(sc, sc_path)
    
    if return_test_set and return_pre_process:
        return test_df, pre_processes
    elif return_test_set:
        return test_df
    elif return_pre_process:
        return pre_processes
    else: return None

def evaluate(config):
    data_path = os.path.join(config["dataset"]["path"], config["dataset"]["name"])
    df, features_dic = open_dataset(data_path, config["dataset"]["name"]+'_test', config["dataset"]["name"])
    
    tester = Tester(df.reset_index(drop=True), features_dic)
    
    encoders = get_encoders(config['model']['encoder'])
    
    device = get_device()
    tester.set_device(device)

    pre_processes_paths = []    
    if not tester.is_clf:
        pre_processes_paths = glob.glob(os.path.join(config['dataset']['pre_process_path'], config['dataset']['name'],'pre_process_label_?.gz'))
        pre_processes_paths = [joblib.load(p) for p in pre_processes_paths]
    
    model_paths = glob.glob(os.path.join(config['model']['path'],config['model']['name']+'*'))
    result = {enc.__str__(): [] for enc in encoders} 
    for i, p in enumerate(model_paths):
        tester.set_model_path(p)
        if len(pre_processes_paths) != 0:
            tester.process_continuous_data(pre_processes_paths[i], 'label')
        for encoder in encoders:
            tester.set_ord_enc(encoder)
            cur_res = tester.test(config['parameters'])
            result[encoder.__str__()].append(cur_res)
    
    for k, v in result.items():
        temp = utility.concat_dic(*v)
        result[k] = utility.mean_std_dic(temp)
    
    return result, tester
        