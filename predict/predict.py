import argparse
import os
import pandas as pd
import numpy as np
import copy
import json
import gzip
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


def process_cat(data, categorical_features):
    if categorical_features:
        data[categorical_features] = data[categorical_features].apply(lambda x: pd.factorize(x)[0])
    return data

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


def get_meta_features(meta_features, model_path, task):
    if task == 'binary_classification':
        cols = ['f%d' % i for i in range(11)]
    elif task == 'regression':
        cols = ['f%d' % i for i in range(9)]
    df = pd.DataFrame.from_dict(meta_features).T
    df.columns = cols
    data_mean_std = pd.read_csv(os.path.join(model_path, 'mean_std_LTE.csv'))
    data_mean_std.columns = ['stats'] + cols
    data_mean_std.set_index('stats', inplace=True)
    df = df.fillna(0)
    df = (df - data_mean_std.loc['mean']) / data_mean_std.loc['std']

    return df


def get_LTE_result(meta_features, model_path, logger, task):
    fi_res = {}
    meta_features = get_meta_features(meta_features, model_path, task)
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


def load_data(eval_file_path):
    def merge_gzip_files(output_file):
        part_files = sorted(os.listdir(eval_file_path))
        with gzip.open(output_file, 'wb') as f_out:
            for part_file in part_files:
                if part_file.startswith(csv_name + '_part'):
                    part_path = os.path.join(eval_file_path, part_file)
                    with gzip.open(part_path, 'rb') as f_in:
                        f_out.write(f_in.read())

    for csv_name in ["train_x", "train_y", "valid_x", "valid_y", "test_x", "test_y"]:
        output_file = os.path.join(eval_file_path, "temp.gz")
        merge_gzip_files(output_file)

        with gzip.open(output_file, 'rb') as f_in:
            with open(os.path.join(eval_file_path, f'{csv_name}.csv'), 'wb') as f_out:
                for line in f_in:
                    f_out.write(line)
        os.remove(output_file)


def run(rank):
    logger.log("Start running file %s" % file_name)
    model_path = os.path.join(directory, 'models/' + model_dir)
    if task == 'binary_classification':
        file_path = os.path.join(directory, 'data/test_data/binary_classification/' + file_name)
    elif task == 'regression':
        file_path = os.path.join(directory, 'data/test_data/regression/' + file_name)
    eval_file_path = os.path.join(file_path, file_name + '_eval_%d' % rank)

    logger.log("Loading data, please wait.")
    load_data(eval_file_path)

    # load data
    train_x = pd.read_csv(os.path.join(eval_file_path, 'train_x.csv'))
    train_y = pd.read_csv(os.path.join(eval_file_path, 'train_y.csv'))
    valid_x = pd.read_csv(os.path.join(eval_file_path, 'valid_x.csv'))
    valid_y = pd.read_csv(os.path.join(eval_file_path, 'valid_y.csv'))
    categorical_features = list(train_x.select_dtypes(exclude=np.number).columns)
    numerical_features = list(train_x.select_dtypes(include=np.number).columns)
    train_x = process_cat(train_x, categorical_features)
    valid_x = process_cat(valid_x, categorical_features)

    data_x = pd.concat([train_x, valid_x])
    data_y = pd.concat([train_y, valid_y])
    logger.log("Successfully load the data.")
    if task == 'binary_classification':
        meta_features = get_meta_features_from_csv_clf(data_x, data_y, numerical_features, categorical_features, logger, n_jobs)
    elif task == 'regression':
        meta_features = get_meta_features_from_csv_reg(data_x, data_y, numerical_features, categorical_features, logger, n_jobs)
    logger.log("Successfully get the meta features of file %s in trial %d" % (file_name, rank))
    with open(os.path.join(eval_file_path, 'meta_features_LTE.json'), 'w') as f:
        json.dump(meta_features, f)
    logger.log("Successfully save the meta features into file %s" % os.path.join(eval_file_path, 'meta_features_LTE.json'))
    lte_result = get_LTE_result(meta_features, model_path, logger, task)
    logger.log("Successfully get the LTE result of file %s in trial %d" % (file_name, rank))
    lte_result.to_csv(os.path.join(eval_file_path, "lte_FI_result.csv"), index=False)
    logger.log("Successfully save the LTE result into file %s" % os.path.join(eval_file_path, "lte_FI_result.csv"))


if __name__ == '__main__':
    from generate_meta_features_clf import get_meta_features_from_csv_clf
    from generate_meta_features_reg import get_meta_features_from_csv_reg

    logger = Logger("predict_LTE")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="directory of FeatureLTE", type=str, default="FeatureLTE")
    parser.add_argument("-f", "--file_name", help="file name of test data", type=str, default="[UCI]Arrhythmia")
    parser.add_argument("-t", "--task", help="binary_classification or regression", type=str, choices=["binary_classification", "regression"], default="binary_classification")
    parser.add_argument("-m", "--model_dir", help="model directory", type=str, default="LTE_models_clf")
    parser.add_argument("-n", "--n_jobs", help="evaluation type", type=int, default="-1")

    args = parser.parse_args()
    file_name = args.file_name
    task = args.task
    directory = args.directory
    model_dir = args.model_dir
    n_jobs = args.n_jobs

    from concurrent.futures import ProcessPoolExecutor

    ex = ProcessPoolExecutor(5)
    futures = []
    try:
        for rank in range(5):
            logger.log("rank: %d" % rank)
            future = ex.submit(run, rank)
            futures.append(future)
        ex.shutdown(wait=True)
        for future in futures:
            try:
                result = future.result()
            except Exception as e:
                logger.log(e)
                import traceback

                traceback.print_exc()
    except Exception:
        import traceback

        logger.log(traceback.format_exc())
