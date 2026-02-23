import os
import pickle as pkl
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef, balanced_accuracy_score

from icscm import InvariantCausalSCM
from scm import SetCoveringMachine

def initiate_model(algo):
    if algo == 'ICSCM':
        model = InvariantCausalSCM()
        add_env = True
    elif algo == 'SCM':
        model = SetCoveringMachine()
        add_env = False
    return model, add_env

def load_dataset(dataset_name):
    df = pd.read_csv(f'datasets/{dataset_name}.csv')
    assert list(df.columns)[0] == 'env'
    assert list(df.columns)[-1] == 'y'
    n_nans = df.isna().sum().sum()
    assert n_nans == 0
    return df

def get_features_used(model, algo):
    features_used = {}
    if algo in ['SCM', 'ICSCM']:
        if hasattr(model, 'rule_importances'):
            for i in range(len(model.rule_importances)):
                if model.rule_importances[i] > 0:
                    features_used[model.model_.rules[i].feature_idx] = model.rule_importances[i]
    else:
        print('algo not recognized')
        print(model)
    return features_used

def get_scores_dict(y_train, pred_train, y_test, pred_test):
    scores_dict = {}
    scores_dict['train'] = {}
    scores_dict['test'] = {}
    scores_dict['train']['accuracy'] = accuracy_score(y_train, pred_train)
    scores_dict['test']['accuracy'] = accuracy_score(y_test, pred_test)
    scores_dict['train']['f1'] = f1_score(y_train, pred_train)
    scores_dict['test']['f1'] = f1_score(y_test, pred_test)
    scores_dict['train']['roc_auc'] = roc_auc_score(y_train, pred_train)
    scores_dict['test']['roc_auc'] = roc_auc_score(y_test, pred_test)
    scores_dict['train']['precision'] = precision_score(y_train, pred_train)
    scores_dict['test']['precision'] = precision_score(y_test, pred_test)
    scores_dict['train']['recall'] = recall_score(y_train, pred_train)
    scores_dict['test']['recall'] = recall_score(y_test, pred_test)
    scores_dict['train']['mcc'] = matthews_corrcoef(y_train, pred_train)
    scores_dict['test']['mcc'] = matthews_corrcoef(y_test, pred_test)
    scores_dict['train']['balanced_accuracy'] = balanced_accuracy_score(y_train, pred_train)
    scores_dict['test']['balanced_accuracy'] = balanced_accuracy_score(y_test, pred_test)
    return scores_dict

def run_exp(dataset_name, algo, split_id, split_proportion=0.5, subsplit_id=None):
    pattern = f'dataset_{dataset_name}_algo_{algo}_split_{split_id}'
    if subsplit_id is not None:
        pattern = pattern + f'_subsplit_{subsplit_id}'
    logs_filepath = f'logs/{pattern}.pkl'
    if os.path.exists(logs_filepath):
        print(f'already done {pattern}')
        with open(logs_filepath, 'rb') as f:
            results = pkl.load(f)
    else:
        print(f'running {pattern}')
        data_df = load_dataset(dataset_name)
        model, add_env = initiate_model(algo)
        # split data
        train_df = data_df.sample(frac=split_proportion, replace=False, axis=0, random_state=split_id)
        val_df = data_df.drop(train_df.index)
        if subsplit_id is not None:
            train_df = train_df.sample(frac=1.0, replace=True, axis=0, random_state=subsplit_id)
        print('train_df.shape', train_df.shape)
        print('env quantities', train_df['env'].value_counts())
        print('env proportions', train_df['env'].value_counts(normalize=True))
        print('y quantities', train_df['y'].value_counts())
        print('y proportions', train_df['y'].value_counts(normalize=True))
        if not add_env:
            train_df = train_df.drop('env', axis=1)
            val_df = val_df.drop('env', axis=1)
        assert train_df.shape[0] + val_df.shape[0] == data_df.shape[0]
        assert train_df.shape[1] == val_df.shape[1]
        # get X and y
        y_train = train_df['y']
        y_val = val_df['y']
        X_train = train_df.drop('y', axis=1)
        X_val = val_df.drop('y', axis=1)
        features_names = list(X_train.columns)
        # assert no nans, convert all to float values, and convert to numpy
        X_train = X_train.astype(float).values
        X_val = X_val.astype(float).values
        y_train = y_train.astype(int).values
        y_val = y_val.astype(int).values
        # 
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        features_used = get_features_used(model, algo)
        scores_dict = get_scores_dict(y_train, pred_train, y_val, pred_val)
        results = {'model': model, 'features used': features_used, 'scores_dict': scores_dict, 'features_names': features_names}
        with open(logs_filepath, 'wb') as f:
            pkl.dump(results, f)
    return results