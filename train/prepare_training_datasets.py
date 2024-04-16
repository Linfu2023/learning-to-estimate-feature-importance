import argparse
import os
import pandas as pd
import numpy as np
import sys
import json

sys.path.append("..")
from logger import Logger
from predict.generate_meta_features_clf import get_meta_features_from_csv_clf
from predict.generate_meta_features_reg import get_meta_features_from_csv_reg


def get_meta_features(data_x, data_y, logger, task, n_jobs):
    categorical_features = list(data_x.select_dtypes(exclude=np.number).columns)
    numerical_features = list(data_x.select_dtypes(include=np.number).columns)
    if task == 'binary_classification':
        meta_features = get_meta_features_from_csv_clf(data_x, data_y, numerical_features, categorical_features, logger, n_jobs)
        cols = ['f%d' % i for i in range(11)]
        label_prefix = 'permutation_importance_auc_v%d.csv'
    elif task == 'regression':
        meta_features = get_meta_features_from_csv_reg(data_x, data_y, numerical_features, categorical_features, logger, n_jobs)
        cols = ['f%d' % i for i in range(9)]
        label_prefix = 'permutation_importance_mape_v%d.csv'

    df = pd.DataFrame.from_dict(meta_features).T
    df.columns = cols
    return df, label_prefix


def run():
    file_list = os.listdir(file_path)
    for file_name in file_list:
        logger.log("Start running file %s" % file_name)

        # load data
        data_x = pd.read_csv(os.path.join(file_path, 'data_x.csv'))
        data_y = pd.read_csv(os.path.join(file_path, 'data_y.csv'))
        logger.log("Successfully load the data.")

        meta_features, label_prefix = get_meta_features(data_x, data_y, logger, task, n_jobs)
        logger.log("Successfully saved the meta features of file %s" % file_name)

        for label_version in range(1, 6):
            with open(os.path.join(file_path, label_prefix % label_version), 'r') as f:
                label = json.load(f)
                mean_tmp = []
                for k, v in label.items():
                    mean_tmp.append(sum(v) / float(len(v)))
            meta_features['label'] = mean_tmp
            meta_features.to_csv("meta_features_LTE_v%d.csv" % label_version, index=False)


if __name__ == '__main__':

    logger = Logger("prepare_training_data")
    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--directory", help="directory of FeatureLTE", type=str, default="FeatureLTE")
    parser.add_argument("-f", "--file_path", help="file folder path of data", type=str, default="FeatureLTE/data/public_datasets/training_datasets_classification")
    parser.add_argument("-t", "--task", help="binary_classification or regression", type=str, choices=["binary_classification", "regression"], default="binary_classification")
    parser.add_argument("-n", "--n_jobs", help="evaluation type", type=int, default="-1")

    args = parser.parse_args()
    file_path = args.file_path
    file_name = os.path.basename(file_path)
    task = args.task
    # directory = args.directory
    n_jobs = args.n_jobs

    try:
        run()
    except Exception:
        import traceback

        logger.log(traceback.format_exc())
