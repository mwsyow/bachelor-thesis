import pandas as pd
import numpy as np
import math
import random
from .encoder import OneHotEncoder, ThermometerEncoder, IntegerEncoder
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone



def df_to_dict(df):
    dic = {}
    for column_name, data in df.items():
        dic[column_name] = data.unique()
    return dic

def concat(enc):
    """_summary_

    Args:
        enc (dictionary): dictionary of feature names and encodings (np.array with shape(num_data, num_unique_values))

    Returns:
        np.array: concatenated encodings (np.array with shape(num_data, sum of each num_unique_values))
    """
    return np.concatenate(list(enc.values()), axis=1)

def concat_dic(*dic_list):
    res_dic = dic_list[0].copy()
    for key in res_dic.keys():
        temp_res = []
        for dic in dic_list:
            temp_res.append(dic[key])
        res_dic[key] = concat_dic(*temp_res) if isinstance(res_dic[key], dict) else temp_res
    
    return res_dic
    
def mean_std_dic(dic, only_mean=False):
    res_dic = dic
    for key, val in res_dic.items():
        if only_mean:
            res_dic[key] = mean_std_dic(val, only_mean) if isinstance(val, dict) else np.mean(val)
        else: res_dic[key] = mean_std_dic(val, only_mean) if isinstance(val, dict) else (np.mean(val), np.std(val))
    return res_dic

def order(features_or, *names):
    features = features_or.copy()
    for name in names:
        features[name] = np.sort(features[name])
    return features


def split_dataset(df, test_size, rand):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    test_index = rand.sample(range(len(df)),math.ceil(test_size * len(df)))
    train_index = list(set(range(len(df))) - set(test_index))
    return X.iloc[train_index], pd.DataFrame(y.iloc[train_index]), X.iloc[test_index], pd.DataFrame(y.iloc[test_index])



def unique_feature_values(df):
    unique_dic = {}
    for col in df.columns:
        unique_dic[col] = df[col].unique()
    return unique_dic


def create_coef_dic(start_idx, ordinal_dic, ordinal_features, *coef_list):
    coef_dic = {}
    idx = start_idx
    for key in ordinal_features:
        unique_val = len(ordinal_dic[key])
        coef_dic[key] = np.stack([np.array(coef_).squeeze()[idx: idx+unique_val] for coef_ in coef_list], axis=0)
        idx += unique_val
    return coef_dic

def create_key_dic(keys, dics, names):
    res_dic = {}
    for key in keys:
        res_dic[key] = {}
        for name, dic in zip(names, dics):
            temp_arr = np.array([])
            for _, v in dic.items():
                temp_arr = np.vstack((temp_arr, v[key])) if temp_arr.size > 0 else v[key]
            res_dic[key][name] = temp_arr
    return res_dic

def round_decimal_places(df, d):
    dec = math.pow(10,d)
    df_copy = df.copy()
    df_copy = df_copy.map(lambda x: (math.ceil(x[0]*dec)/dec, math.ceil(x[1]*dec)/dec))
    return df_copy

def encoding_pipeline(X_train, X_test, encoder, dic, step_sizes=None):
    cat_encoder = encoder(dic, step_sizes)
    train_dic, test_dic = cat_encoder(X_train), cat_encoder(X_test)
    cat_train, cat_test = concat(train_dic), concat(test_dic)
    return cat_train, cat_test, cat_encoder


def train_and_predict(encoders, regr, X_ordinal_train, X_ordinal_test, ordinal_dic, label_train, preprocess=None, step_sizes=None, other_cat=None, other_num=None):
    res_dic = {'train':{}, 'test':{}}
    res_model = {}
    train_all, test_all = [], []
    
    if not other_num is None:
        if 'train' in other_num and 'test' in other_num:
            if not preprocess is None :
                preprocess.fit(other_num['train'])
                other_num['train'] = preprocess.transform(other_num['train'])
                other_num['test'] = preprocess.transform(other_num['test'])
            train_all += [other_num['train']]
            test_all += [other_num['test']]
        
    if not other_cat is None:
        if 'train' in other_cat and 'test' in other_cat:
            train_all += [other_cat['train']]
            test_all += [other_cat['test']]
            
    for encoder in encoders:
    
        ordinal_train, ordinal_test, _ = encoding_pipeline(X_ordinal_train, X_ordinal_test, encoder, ordinal_dic, step_sizes)
        
        X_train_all = np.concatenate(train_all + [ordinal_train], axis=1)
        X_test_all = np.concatenate(test_all + [ordinal_test], axis=1)
        
        trained_model = clone(regr)
        trained_model.fit(X_train_all, label_train)
        y_pred = trained_model.predict(X_test_all)
        y_pred_train = trained_model.predict(X_train_all)

        res_dic['test'][encoder.__str__()] = y_pred
        res_dic['train'][encoder.__str__()] = y_pred_train
        res_model[encoder.__str__()] = trained_model
        
    return res_dic, res_model


def k_fold_validation(k, df, encoders, pred_task, features, step_sizes=None, return_model=False, train_result=False):
    
    assert pred_task['name'] in ['classification', 'regression'], 'unknown prediction task'
    
    final_result = {} 
    final_result_train = {} 
    result_models = {}
    result_all = {'true': []}
    result_all_train = {'true': []}
    for encoder in encoders:
        result_all[encoder.__str__()] = []
        result_all_train[encoder.__str__()] = []
        final_result[encoder.__str__()] = {}
        final_result_train[encoder.__str__()] = {}
        result_models[encoder.__str__()] = []
        
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    kf = StratifiedKFold(k)    
    
    for train_index, test_index in kf.split(df, y):  
        X_train, y_train, X_test, y_test = X.iloc[train_index], pd.DataFrame(y.iloc[train_index]), X.iloc[test_index], pd.DataFrame(y.iloc[test_index])
        other_cat, other_num = None, None
        if 'nominal' in features:
            if not features['nominal'] is None:   
                nominal_train, nominal_test, _ = encoding_pipeline(X_train[features['nominal']['name']], X_test[features['nominal']['name']], OneHotEncoder, features['nominal']['dict'])
                other_cat = {'train': nominal_train, 'test': nominal_test}
        if 'numerical' in features:
            if not features['numerical'] is None:
                other_train, other_test = np.array(X_train[features['numerical']].values), np.array(X_test[features['numerical']].values)
                other_num = {'train': other_train, 'test': other_test}
        if pred_task['name'] == 'classification' and 'label' in features:
            y_train, y_test, _ = encoding_pipeline(y_train, y_test, IntegerEncoder, features['label'])
        elif pred_task['name'] == 'regression':
            sc = StandardScaler()
            y_train, y_test = sc.fit_transform(y_train), sc.transform(y_test)


        result, models = train_and_predict(
                encoders, 
                pred_task['model'], 
                X_train[features['ordinal']['name']], X_test[features['ordinal']['name']], features['ordinal']['dict'], 
                y_train, 
                preprocess=StandardScaler(),
                step_sizes=step_sizes,
                other_cat=other_cat,
                other_num=other_num
        )
        result_all['true'] = result_all['true'] + [np.squeeze(y_test)] if len(result_all['true'])> 0 else [np.squeeze(y_test)]
        result_all_train['true'] = result_all_train['true'] + [np.squeeze(y_train)] if len(result_all_train['true'])> 0 else [np.squeeze(y_train)]
        
        for encoder in encoders:
            if train_result:
                result_all_train[encoder.__str__()] = result_all_train[encoder.__str__()] + [np.squeeze(result['train'][encoder.__str__()])] if len(result_all_train[encoder.__str__()])> 0 else [np.squeeze(result['train'][encoder.__str__()])]
            result_all[encoder.__str__()] = result_all[encoder.__str__()] + [np.squeeze(result['test'][encoder.__str__()])] if len(result_all[encoder.__str__()])> 0 else [np.squeeze(result['test'][encoder.__str__()])]
            result_models[encoder.__str__()].append(models[encoder.__str__()]) 
    for encoder in encoders:
        for name, metric in pred_task['metric'].items():
            temp_result = []
            temp_result_train = []
            for i in range(k):
                if train_result:
                    temp_result_train.append(metric(result_all_train['true'][i], (result_all_train[encoder.__str__()][i]))) 
                temp_result.append(metric(result_all['true'][i], (result_all[encoder.__str__()][i]))) 
            if pred_task['name'] == 'classification' and name == 'clf_rep':
                if train_result:
                    clf_rep_dic = concat_dic(*temp_result_train)
                    final_result_train[encoder.__str__()][name] = mean_std_dic(clf_rep_dic)
                clf_rep_dic = concat_dic(*temp_result)
                final_result[encoder.__str__()][name] = mean_std_dic(clf_rep_dic)
            else: 
                if train_result:
                    final_result_train[encoder.__str__()][name] = (np.mean(temp_result_train), np.std(temp_result_train))
                final_result[encoder.__str__()][name] = (np.mean(temp_result), np.std(temp_result))
    
    return_values = [final_result]
    if return_model:
        return_values.append(result_models)
    if train_result:
        return_values.append(final_result_train)
    return tuple(return_values)
    