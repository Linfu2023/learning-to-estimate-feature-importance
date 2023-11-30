import ray.internal
import lightgbm as lgb
import fasttreeshap
import logging

import os
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import correlation
import random
import ray
from sklearn.feature_selection import f_classif, f_regression
from sklearn.metrics import mutual_info_score
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
from statsmodels.stats.oneway import anova_oneway


@ray.remote
def calculate_mi_and_chi2(data, label, feature, flag):
    x = data[feature].values
    y = label.values.ravel()
    if flag:
        tab = pd.crosstab(y, x)
        mi_score = mutual_info_score(y, x, contingency=tab)
        chi2_p = chi2_contingency(tab)
        chi2_p = 1 - chi2_p[1]
    else:
        x_bin = pd.cut(x, 20).codes
        tab = pd.crosstab(y, x_bin)
        mi_score = mutual_info_score(y, x_bin, contingency=tab)
        chi2_p = -1
    return mi_score, chi2_p


def get_mutual_info_and_chi2(features, data_id, label_id, features_type, value_counts):
    mi_dict = dict()
    chi2_p_value = dict()
    results = dict()
    for feature in features:
        if features_type[feature] == 'categorical' or (1 < value_counts[feature] <= 20):
            results[feature] = calculate_mi_and_chi2.remote(data_id, label_id, feature, 1)
        else:
            results[feature] = calculate_mi_and_chi2.remote(data_id, label_id, feature, 0)

    for feature in features:
        result = ray.get(results[feature])
        mi_dict[feature] = result[0]
        chi2_p_value[feature] = result[1]
    return mi_dict, chi2_p_value


@ray.remote
def calculate_mean(data_id, feature):
    return data_id[feature].mean()


def get_mean(features, data_id):
    mean_list = []
    for feature in features:
        mean_list.append(calculate_mean.remote(data_id, feature))
    results = ray.get(mean_list)
    return np.array(results)


@ray.remote
def calculate_std(data_id, feature):
    return data_id[feature].std()


def get_std(features, data_id):
    std_list = []
    for feature in features:
        std_list.append(calculate_std.remote(data_id, feature))
    results = ray.get(std_list)
    return np.array(results)


@ray.remote
def calculate_anova_p_value(data, label, feature):
    import warnings
    warnings.filterwarnings('ignore')

    x = data[feature]
    y = label.values.ravel()
    f_value, p_value = f_classif(x.values.reshape(-1, 1), y)
    welch_anova_result = anova_oneway(x.values, y, use_var='unequal', welch_correction=True)
    return feature, p_value[0], welch_anova_result[1]


def get_anova_p_value(features, data_id, label_id):
    p_value_list = []
    for feature in features:
        p_value_list.append(calculate_anova_p_value.remote(data_id, label_id, feature))
    results = ray.get(p_value_list)
    anova_p_values = dict()
    welch_anova_p_values = dict()
    for feature, anova_p, welch_anova_p in results:
        anova_p_values[feature] = 1 - anova_p
        welch_anova_p_values[feature] = 1 - welch_anova_p
    return anova_p_values, welch_anova_p_values


@ray.remote
def calculate_anova_p_value_reg(data, label, feature):
    import warnings
    warnings.filterwarnings('ignore')

    x = data[feature]
    y = label.values.ravel()
    f_value, p_value = f_regression(x.values.reshape(-1, 1), y)
    # welch_anova_result = anova_oneway(x.values, y, use_var='unequal', welch_correction=True)
    welch_anova_result = anova_oneway(data=[x.values, y], use_var='unequal', welch_correction=True)
    return feature, p_value[0], welch_anova_result[1]


def get_anova_p_value_reg(features, data_id, label_id):
    p_value_list = []
    for feature in features:
        p_value_list.append(calculate_anova_p_value_reg.remote(data_id, label_id, feature))
    results = ray.get(p_value_list)
    anova_p_values = dict()
    welch_anova_p_values = dict()
    for feature, anova_p, welch_anova_p in results:
        anova_p_values[feature] = 1 - anova_p
        welch_anova_p_values[feature] = 1 - welch_anova_p
    return anova_p_values, welch_anova_p_values


def fit(X, y, y_mean, cols=None):
    if cols is None:
        cols = X.columns.values
    results = {}
    for col in cols:
        results[col] = fit_column_map.remote(X, y, col, y_mean)
    for col in cols:
        results[col] = ray.get(results[col])
    return results


@ray.remote
def fit_column_map(X, y, col, y_mean):
    series = X[col]
    category = pd.Categorical(series)

    categories = category.categories
    codes = category.codes.copy()

    codes[codes == -1] = len(categories)
    categories = np.append(categories, np.nan)
    return_map = pd.Series(dict([(code, category) for code, category in enumerate(categories)]))

    result = y.groupby(codes).agg(['sum', 'count'])
    result = result.rename(return_map)
    result = result['label']
    result['result'] = (result['sum'] + y_mean.values[0]) / (result['count'] + 1)
    return result


@ray.remote
def _trans(X, y, col, y_mean):
    agg = y.groupby(X[col]).agg(['count', 'mean'])
    agg = agg['label']
    counts = agg['count']
    means = agg['mean']
    result = (counts * means + y_mean) / (counts + 1)
    return [col, result]


def transform(X, y, y_mean, categorical_features):
    """
    The model uses a single column of floats to represent the means of the target variables.
    """

    # Prepare the data

    results = []
    for col in categorical_features:
        # Simulation of CatBoost implementation, which calculates leave-one-out on the fly.
        # The nice thing about this is that it helps to prevent overfitting. The bad thing
        # is that CatBoost uses many iterations over the data. But we run just one iteration.
        # Still, it works better than leave-one-out without any noise.
        # See:
        #   https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/
        # Cumsum does not work nicely with None (while cumcount does).
        # As a workaround, we cast the grouping column as string.
        # See: issue #209
        results.append(_trans.remote(X, y, col, y_mean))
    results = ray.get(results)

    return results


def cat2num_transform(data, label, y_mean, categorical_features, version=1):
    label = label.astype('double')
    data_ray = ray.put(data)
    label_ray = ray.put(label)
    if version == 1:
        results = transform(data_ray, label_ray, y_mean, categorical_features)
        for col, result in results:
            data[col] = data[col].map(result)

    if version == 2:
        mapping = fit(
            data_ray, label_ray, y_mean,
            cols=categorical_features
        )
        for col in categorical_features:
            data[col] = np.array(data[col].map(mapping[col]['result']))

    ray.get(data_ray)
    ray.get(label_ray)
    ray.internal.free(data_ray)
    ray.internal.free(label_ray)
    return data


assert ray.__version__ == '1.0.1.post1'
assert lgb.__version__ == '3.3.2'


def get_gain_importance(data, label, n_jobs):
    gbm = lgb.LGBMClassifier(**{'importance_type': 'gain', 'seed': 0, 'deterministic': True,
                                'n_jobs': n_jobs, 'force_row_wise': True})
    gbm.fit(data.values, label.values.ravel())
    FI = get_ranking(np.log(gbm.feature_importances_ + 1))
    gain_importance = {}
    for i, column in enumerate(data.columns):
        gain_importance[column] = FI[i]
    return gain_importance


def get_minority_percent(label):
    n_instance = len(label)
    value_counts = label[label.columns[0]].value_counts()
    minority_percent = value_counts.min() / n_instance
    return minority_percent, n_instance


def get_ranking(score_list):
    score_list = np.asarray(score_list)
    maximum = np.nanmax(score_list)
    if maximum == 0:
        return score_list
    score_list = score_list / maximum
    return list(score_list)


def process_cat(data, categorical_features):
    logging.info('Start to process_cat...')
    data[categorical_features] = data[categorical_features].apply(lambda x: pd.factorize(x)[0])
    return data


def preprocess_data(numerical_features, categorical_features):
    if numerical_features != '':
        numerical_features = numerical_features.split(',')
    else:
        numerical_features = []
    if categorical_features != '':
        categorical_features = categorical_features.split(',')
    else:
        categorical_features = []
    all_features = sorted(numerical_features + categorical_features)
    features_type = {}
    for feature in all_features:
        if feature in categorical_features:
            features_type[feature] = 'categorical'
        else:
            features_type[feature] = 'continuous'
    return features_type, all_features, numerical_features, categorical_features


def get_distance_correlation(data, label):
    dis_series = {}
    for col_name, values in data.iteritems():
        dis_series[col_name] = correlation(values, label.values.ravel())
    return dis_series


def get_shap_score(data, label, n_jobs):
    gbm = lgb.LGBMClassifier(**{'importance_type': 'gain', 'seed': 0, 'deterministic': True,
                                'n_jobs': n_jobs, 'force_row_wise': True})
    gbm.fit(data.values, label.values.ravel())
    explainer = fasttreeshap.TreeExplainer(gbm)
    shap_values = np.mean(np.abs(explainer.shap_values(data.values)[0]), 0)
    n_shap_feat_imp = get_ranking(np.log(shap_values + 1))
    shap_score = dict()
    for i, column in enumerate(data.columns):
        shap_score[column] = n_shap_feat_imp[i]
    return shap_score


def get_fasttreeshap_importance(data, label, n_jobs):
    gbm = lgb.LGBMClassifier(**{'importance_type': 'gain', 'seed': 0, 'deterministic': True,
                                'n_jobs': n_jobs, 'force_row_wise': True})
    gbm.fit(data.values, label.values.ravel())

    explainer = fasttreeshap.TreeExplainer(gbm, algorithm="v1", n_jobs=n_jobs)

    shap_values = np.mean(np.abs(np.array(explainer.shap_values(data.values))), 0)
    n_shap_feat_imp = get_ranking(np.log(shap_values + 1))
    shap_score = dict()
    for i, column in enumerate(data.columns):
        shap_score[column] = n_shap_feat_imp[i]
    return shap_score


# train a Lasso model by ADMM
def get_lasso_coef(data, label, alpha=0.001, max_iter=1000, rho=1):
    X = data
    m, n = X.shape
    threshold = alpha / rho
    inverse_mat = np.linalg.inv(np.dot(X.T, X) / m + rho * np.identity(n))
    mat_temp = np.dot(X.T, label) / m
    theta = mat_temp.copy()
    mu = np.zeros(n)
    omega = np.dot(inverse_mat, mat_temp + rho * theta - mu)

    def _soft_thresholding_func(x, threshold):
        pos_index = (x > threshold)
        neg_index = (x < threshold)
        zero_index = (abs(x) <= threshold)

        y = np.zeros(x.shape)

        y[pos_index] = x[pos_index] - threshold
        y[neg_index] = x[neg_index] + threshold
        y[zero_index] = 0.0

        return y

    for _ in range(max_iter):
        omega = np.dot(inverse_mat, mat_temp + rho * theta - mu)
        theta = _soft_thresholding_func(omega + mu / rho, threshold)

        mu = mu + rho * (omega - theta)

    return omega


def get_lasso_score(data, data_normalized, label, alpha=0.001, max_iter=1000):
    omega = get_lasso_coef(data_normalized, label, alpha, max_iter)
    omega = np.where(np.abs(omega) < 1e-10, 0, omega)
    coef_scores = get_ranking(np.abs(omega))
    lasso_score = dict(zip(data.columns, coef_scores))

    return lasso_score


def cat2num(data, label, categorical_features):
    y_mean = label.mean().values[0]
    if categorical_features:
        data = cat2num_transform(data, label, y_mean, categorical_features, version=1)
    return data


def get_value_counts(data):
    value_counts = dict()
    for column in data.columns:
        value_counts[column] = len(data[column].value_counts())
    return value_counts


def get_mean_by_std(data, label):
    data_temp = data.copy()
    data_temp = data_temp / data_temp.apply(lambda x: max(-x.min(), x.max()))
    data_temp_pos = data_temp.loc[label[label.columns[0]] == 1]
    data_temp_neg = data_temp.loc[label[label.columns[0]] == 0]
    data_temp_pos_ray = ray.put(data_temp_pos)
    data_temp_neg_ray = ray.put(data_temp_neg)
    mean_pos = get_mean(data.columns, data_temp_pos_ray)
    mean_neg = get_mean(data.columns, data_temp_neg_ray)
    std_pos = get_std(data.columns, data_temp_pos_ray)
    std_neg = get_std(data.columns, data_temp_neg_ray)
    mean_by_std_array = np.abs((mean_pos - mean_neg) / (std_pos + std_neg))
    mean_by_std = dict()
    for i, feature in enumerate(data.columns):
        mean_by_std[feature] = mean_by_std_array[i]
    ray.get(data_temp_pos_ray)
    ray.internal.free(data_temp_pos_ray)
    ray.get(data_temp_neg_ray)
    ray.internal.free(data_temp_neg_ray)
    return mean_by_std


def run_calculation_simple(data, label, numerical_features, categorical_features, logger, n_jobs=8):
    random.seed(0)
    np.random.seed(0)
    os.environ['PYTHONHASHSEED'] = '0'
    # try:
    label.columns = ['label']
    features_type, all_features, numerical_features, categorical_features \
        = preprocess_data(numerical_features, categorical_features)


    logger.log('Start to initialize ray...')
    logger.log("n_jobs: %d" % n_jobs)
    ray.init(num_cpus=n_jobs, include_dashboard=False)
    logger.log('ray initialized.')

    logger.log('Start to cat2num...')
    data = cat2num(data, label, categorical_features)

    # if categorical_features != '':
    #     data = process_cat(data, categorical_features)
    logger.log('Start to calc gain_importance...')
    gain_importance = get_gain_importance(data, label, n_jobs)
    logger.log('Start to calc shap_importance...')
    shap_score = get_fasttreeshap_importance(data, label, n_jobs)
    minority_percent, n_instance = get_minority_percent(label)

    value_counts = get_value_counts(data)
    data[numerical_features] = data[numerical_features].fillna(0)

    # use ray to calculate
    data_ray = ray.put(data)
    label_ray = ray.put(label)

    logger.log('Start to chi2...')
    mi_dict, chi2_p_value = get_mutual_info_and_chi2(data.columns, data_ray, label_ray, features_type, value_counts)

    logger.log('Start to calc anova...')
    anova_p_value, welch_anova_p_value = get_anova_p_value(data.columns, data_ray, label_ray)

    ray.get(data_ray)
    ray.internal.free(data_ray)

    logger.log('Start to calc distance_correlation...')
    distance_correlation = get_distance_correlation(data, label)

    logger.log('Start to calc mean & std...')
    mean_by_std = get_mean_by_std(data, label)

    logger.log('Start to calc lasso score...')
    data_normalized = StandardScaler().fit_transform(data)
    lasso_score = get_lasso_score(data, data_normalized, label)

    meta_features_dict_simple = {}
    for i in range(len(all_features)):
        meta_features = []
        feature_name = all_features[i]

        # useless meta-features
        meta_features.append(minority_percent)

        # attributes of the feature
        if features_type[feature_name] == 'categorical':
            meta_features.append(0)
        else:
            meta_features.append(1)

        # 15
        # filters methods
        meta_features.append(chi2_p_value[feature_name])
        meta_features.append(mi_dict[feature_name])
        meta_features.append(anova_p_value[feature_name])
        meta_features.append(welch_anova_p_value[feature_name])
        meta_features.append(mean_by_std[feature_name])

        # model based methods
        meta_features.append(lasso_score[feature_name][0])
        meta_features.append(gain_importance[feature_name])
        meta_features.append(shap_score[feature_name])
        meta_features.append(distance_correlation[feature_name])
        meta_features_dict_simple[feature_name] = meta_features
    ray.shutdown()
    logger.log('Shut down ray...')
    meta_features = []
    for i, feature_name in enumerate(data.columns):
        meta_features.append(meta_features_dict_simple[feature_name])
    return meta_features, data.columns
    # except Exception as e:
    #     ray.shutdown()


def get_meta_features_from_csv_clf(data_x, data_y, numerical_features, categorical_features, logger, n_jobs=-1):
    if not categorical_features:
        cf = ''
    else:
        cf = ','.join(categorical_features)
    if not numerical_features:
        nf = ''
    else:
        nf = ','.join(numerical_features)
    meta_features, features = run_calculation_simple(data_x, data_y, nf, cf, logger, int(n_jobs))
    mf = dict(zip(features, meta_features))
    return mf
