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
