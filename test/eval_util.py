import numpy as np
import pandas as pd
from copy import deepcopy
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
import fasttreeshap


def get_n_estimators_by_cv(params, dtrain, seed=0):
    validation_summary = lgb.cv(params,
                                dtrain,
                                num_boost_round=1000,  # any high number will do
                                nfold=5,
                                stratified=True,
                                metrics=['auc'],
                                early_stopping_rounds=30,  # Here it is
                                seed=seed,
                                verbose_eval=None)
    optimal_n_estimators = len(validation_summary["auc-mean"])
    best_score = np.max(validation_summary["auc-mean"])
    return optimal_n_estimators, best_score


def get_n_estimators_by_valid(params, dtrain, dvalid):
    bst = lgb.train(params=params,
                    train_set=dtrain,
                    num_boost_round=1000,  # any high number will do
                    valid_sets=[dvalid],
                    early_stopping_rounds=30,
                    verbose_eval=False)
    n_estimators, best_score = bst.best_iteration, bst.best_score.get('valid_0').get('auc')
    return n_estimators, best_score


def gridsearch_tuning(train_x, train_y, valid_x, valid_y, n_jobs):
    dtrain = lgb.Dataset(data=train_x, label=train_y)
    dvalid = lgb.Dataset(data=valid_x, label=valid_y)
    my_params = {
        'objective': 'binary',
        'learning_rate': 0.1,
        'max_depth': 10,
        'verbose': -1,
        'seed': 0,
        'metric': ['auc'],
        'deterministic': True,
        'n_jobs': n_jobs
    }

    num_leaves_choice = [255, 127, 63, 31, 15, 7, 3]
    valid_results = []
    for value in num_leaves_choice:
        my_params_temp = deepcopy(my_params)
        my_params_temp['num_leaves'] = value
        if valid_x is None:
            n_estimators, best_score = get_n_estimators_by_cv(my_params_temp, dtrain)
        else:
            n_estimators, best_score = get_n_estimators_by_valid(my_params_temp, dtrain, dvalid)
        valid_results.append([my_params_temp, n_estimators, best_score])
    valid_results = sorted(valid_results, key=lambda x: x[2])
    best_params = valid_results[-1][0]
    best_params['n_estimators'] = valid_results[-1][1]
    best_score = valid_results[-1][2]
    return best_params, best_score


def selection_according_to_prediction(train_x, valid_x, test_x, fi, percent):
    sorted_fi = sorted(fi.items(), key=lambda item: item[1], reverse=True)
    selected_columns = []
    for column, value in sorted_fi[:int(np.ceil(len(sorted_fi) * percent))]:
        selected_columns.append(column)
    # print(selected_columns)
    return train_x[selected_columns].copy(), valid_x[selected_columns].copy(), test_x[selected_columns].copy()


def get_mdi_default_result(train_x, train_y, valid_x, valid_y, n_jobs=8):
    model = lgb.LGBMClassifier(**{'importance_type': 'gain', 'deterministic': True,
                                  'n_jobs': n_jobs, 'force_row_wise': True})
    model.fit(train_x.values, train_y.values.ravel(), eval_set=[(valid_x.values, valid_y.values.ravel())])
    feat_imp = model.feature_importances_
    mdi = {}
    for i, column in enumerate(train_x.columns):
        mdi[column] = feat_imp[i]
    return mdi


def get_mdi_tuned_result(train_x, train_y, valid_x, valid_y, n_jobs=8):
    best_params, cv_auc = gridsearch_tuning(train_x, train_y, valid_x, valid_y, n_jobs)
    best_params['importance_type'] = 'gain'
    model = lgb.LGBMClassifier(**best_params)
    model.fit(train_x.values, train_y.values.ravel(), eval_set=[(valid_x.values, valid_y.values.ravel())])
    feat_imp = model.feature_importances_
    mdi = {}
    for i, column in enumerate(train_x.columns):
        mdi[column] = feat_imp[i]
    return mdi


def get_shap_default_result(train_x, train_y, valid_x, valid_y, n_jobs=8):
    model = lgb.LGBMClassifier(**{'importance_type': 'gain', 'deterministic': True,
                                  'n_jobs': n_jobs, 'force_row_wise': True})
    model.fit(train_x.values, train_y.values.ravel(), eval_set=[(valid_x.values, valid_y.values.ravel())])
    explainer = fasttreeshap.TreeExplainer(model, algorithm="v1", n_jobs=n_jobs)
    shap_values = np.mean(np.abs(np.array(explainer.shap_values(train_x.values))), 0)
    res = {}
    for i, column in enumerate(train_x.columns):
        res[column] = shap_values[i]
    return res


def get_shap_tuned_result(train_x, train_y, valid_x, valid_y, n_jobs=8):
    best_params, cv_auc = gridsearch_tuning(train_x, train_y, valid_x, valid_y, n_jobs)
    best_params['importance_type'] = 'gain'
    model = lgb.LGBMClassifier(**best_params)
    model.fit(train_x.values, train_y.values.ravel(), eval_set=[(valid_x.values, valid_y.values.ravel())])
    explainer = fasttreeshap.TreeExplainer(model, algorithm="v1", n_jobs=n_jobs)
    shap_values = np.mean(np.abs(np.array(explainer.shap_values(train_x.values))), 0)
    res = {}
    for i, column in enumerate(train_x.columns):
        res[column] = shap_values[i]
    return res


def get_pi_single_result(train_x, train_y, valid_x, valid_y, n_jobs=8):
    best_params, cv_auc = gridsearch_tuning(train_x, train_y, valid_x, valid_y, n_jobs)
    best_params['importance_type'] = 'gain'
    model = lgb.LGBMClassifier(**best_params)
    model.fit(train_x.values, train_y.values.ravel(), eval_set=[(valid_x.values, valid_y.values.ravel())])
    r = permutation_importance(model, valid_x.values, valid_y.values.ravel(), scoring='roc_auc', n_jobs=8)
    r_mean = r.importances_mean
    pi = {}
    for i, column in enumerate(train_x.columns):
        pi[column] = r_mean[i]
    return pi


def get_pi_ensemble_result(train_x, train_y, valid_x, valid_y, n_jobs=8):
    data_x = pd.concat([train_x, valid_x])
    data_y = pd.concat([train_y, valid_y])
    best_params, cv_auc = gridsearch_tuning(data_x, data_y, None, None, n_jobs)
    best_params['importance_type'] = 'gain'
    seed = 1
    skf_random_states = [i + (seed - 1) * 3 for i in [1, 2, 3]]
    permutation_random_states = [i + (seed - 1) * 3 for i in [1, 2, 3]]
    results_dict = {}
    columns = train_x.columns
    for column in columns:
        results_dict[column] = []

    X = data_x.values
    y = data_y.values.ravel()
    for i in range(3):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=skf_random_states[i])
        cv_fold = 0
        n_repeats = 1
        for train_index, test_index in skf.split(X, y):
            cv_fold += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = lgb.LGBMClassifier(**best_params)
            model.fit(X_train, y_train)
            r = permutation_importance(model, X_test, y_test, scoring='roc_auc', n_repeats=n_repeats,
                                       random_state=permutation_random_states[i], n_jobs=8)
            r_mean = r.importances_mean
            for j, column in enumerate(columns):
                results_dict[column].append(r_mean[j])

    pimp_15 = dict()
    for key in results_dict:
        pimp_15[key] = np.mean(results_dict[key])

    return pimp_15


def get_random_result(train_x, train_y, valid_x, valid_y, n_jobs=8):
    random_score = np.random.rand(len(train_x.columns))
    res = {}
    for i, column in enumerate(train_x.columns):
        res[column] = float(random_score[i])
    return res
