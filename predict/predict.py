import argparse
import os
import pandas as pd
import numpy as np
import copy
import json
import sys

sys.path.append("..")
from train.LambdaRank import LambdaRankNN
from train.logger import Logger


def transform_results(pred):
    sorted_pred = copy.deepcopy(pred)
    temp = np.argsort(pred)
    for i in range(len(pred)):
        sorted_pred[temp[i]] = i
    sorted_pred = sorted_pred / (len(pred) - 1)
    return sorted_pred


def predict(meta_features, seed, model_path):
    model1 = LambdaRankNN()
    model1.load(os.path.join(model_path, 'LTE_s%d_v1' % seed))
    model2 = LambdaRankNN()
    model2.load(os.path.join(model_path, 'LTE_s%d_v2' % seed))
    model3 = LambdaRankNN()
    model3.load(os.path.join(model_path, 'LTE_s%d_v3' % seed))
    model4 = LambdaRankNN()
    model4.load(os.path.join(model_path, 'LTE_s%d_v4' % seed))
    model5 = LambdaRankNN()
    model5.load(os.path.join(model_path, 'LTE_s%d_v5' % seed))
    pred1 = transform_results(model1.predict(meta_features))
    pred2 = transform_results(model2.predict(meta_features))
    pred3 = transform_results(model3.predict(meta_features))
    pred4 = transform_results(model4.predict(meta_features))
    pred5 = transform_results(model5.predict(meta_features))
    prediction = (pred1 + pred2 + pred3 + pred4 + pred5) / 5
    return prediction


def get_meta_features(file_path, model_path):
    cols = ['f%d' % i for i in range(11)]
    with open(os.path.join(file_path, 'meta_features_LTE.json'), 'r') as f:
        mf = json.load(f)
    df = pd.DataFrame.from_dict(mf).T
    df.columns = cols
    data_mean_std = pd.read_csv(os.path.join(model_path, 'mean_std_LTE.csv'))
    data_mean_std.columns = ['stats'] + cols
    data_mean_std.set_index('stats', inplace=True)
    df = df.fillna(0)
    df = (df - data_mean_std.loc['mean']) / data_mean_std.loc['std']

    return df


def get_LTE_result(meta_features, model_path):
    fi_res = {}
    for seed in [1, 2, 3, 4, 5]:
        try:
            logger.log("seed: %d" % seed)
            pred = list(map(float, predict(meta_features.values, seed, model_path)))
            feature_importance = dict(zip(meta_features.index, pred))
            fi_res['fi_s%d' % seed] = feature_importance

        except Exception as e:
            logger.log("Failed to get LTE result....")
            import traceback
            logger.log(traceback.format_exc())
    feature_name = list(fi_res['fi_s1'].keys())
    df = pd.DataFrame({'feature_name': feature_name,
                       'V1': list(transform_results(np.array(list(fi_res['fi_s1'].values())))),
                       'V2': list(transform_results(np.array(list(fi_res['fi_s2'].values())))),
                       'V3': list(transform_results(np.array(list(fi_res['fi_s3'].values())))),
                       'V4': list(transform_results(np.array(list(fi_res['fi_s4'].values())))),
                       'V5': list(transform_results(np.array(list(fi_res['fi_s5'].values()))))})
    df['mean'] = np.array((df['V1'] + df['V2'] + df['V3'] + df['V4'] + df['V5']) / 5)
    return df


def run(rank):
    logger.log("Start running file %s" % file_name)
    file_path = os.path.join(directory, 'data/test_data_for_evaluation/' + file_name)
    model_path = os.path.join(directory, 'models/' + model_type)
    eval_file_path = os.path.join(file_path, file_name + '_eval_%d' % rank)
    meta_features = get_meta_features(eval_file_path, model_path)
    logger.log("Successfully load the meta features of file %s in trial %d" % (file_name, rank))
    lte_result = get_LTE_result(meta_features, model_path)
    logger.log("Successfully get the LTE result of file %s in trial %d" % (file_name, rank))
    lte_result.to_csv(os.path.join(eval_file_path, "lte_result.csv"), index=False)


if __name__ == '__main__':

    logger = Logger("predict_LambdaRank_LTE")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="directory of FeatureLTE", type=str, default="FeatureLTE")
    parser.add_argument("-f", "--file_name", help="file name of test data", type=str, default="[UCI]Arrhythmia")
    parser.add_argument("-m", "--model_type", help="model type", type=str, default="LTE_models_clf")

    args = parser.parse_args()
    file_name = args.file_name
    directory = args.directory
    model_type = args.model_type

    from concurrent.futures import ProcessPoolExecutor

    ex = ProcessPoolExecutor(5)
    try:
        for rank in range(5):
            logger.log("rank: %d" % rank)
            ex.submit(run, rank)
        ex.shutdown(wait=True)
    except Exception:
        import traceback

        logger.log(traceback.format_exc())
