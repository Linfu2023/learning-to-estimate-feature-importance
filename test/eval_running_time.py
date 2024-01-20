import os
import argparse
from sklearn.model_selection import train_test_split
import time
import gzip
import json
from eval_utils import *
import sys

sys.path.append("..")
from train.logger import Logger
from predict.predict import get_LTE_result
from reproduce_eval_result import get_meta_features
from plot_utils import plot_fig4


def process_cat(data, categorical_features):
    if categorical_features:
        data[categorical_features] = data[categorical_features].apply(lambda x: pd.factorize(x)[0])
    return data


def load_data(eval_file_path):
    def merge_gzip_files(output_file):
        part_files = sorted(os.listdir(eval_file_path))
        with gzip.open(output_file, 'wb') as f_out:
            for part_file in part_files:
                if part_file.startswith(csv_name + '_part'):
                    part_path = os.path.join(eval_file_path, part_file)
                    with gzip.open(part_path, 'rb') as f_in:
                        f_out.write(f_in.read())

    for csv_name in ["data_x", "data_y"]:
        output_file = os.path.join(eval_file_path, "temp.gz")
        merge_gzip_files(output_file)

        with gzip.open(output_file, 'rb') as f_in:
            with open(os.path.join(eval_file_path, f'{csv_name}.csv'), 'wb') as f_out:
                for line in f_in:
                    f_out.write(line)
        os.remove(output_file)


def multiple_data_to_million(data_path):
    load_data(data_path)
    data_x = pd.read_csv(os.path.join(data_path, "data_x.csv"))
    data_y = pd.read_csv(os.path.join(data_path, "data_y.csv"))
    categorical_features = list(data_x.select_dtypes(exclude=np.number).columns)
    data_x = process_cat(data_x, categorical_features)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=0.2, random_state=0)

    repeats = -(-1000000 // len(data_y))
    train_x_out = pd.concat([train_x] * repeats, ignore_index=True)
    train_y_out = pd.concat([train_y] * repeats, ignore_index=True)
    valid_x_out = pd.concat([valid_x] * repeats, ignore_index=True)
    valid_y_out = pd.concat([valid_y] * repeats, ignore_index=True)

    return train_x_out, train_y_out, valid_x_out, valid_y_out


def run():
    file_path = os.path.join(directory, 'data/test_data/running_time_test')
    if simple:
        eval_list = ["lte", "mdi_default", "mdi_tuned", "shap_default"]
    else:
        eval_list = ["lte", "mdi_default", "mdi_tuned", "shap_default", "shap_tuned", "pi_single", "pi_ensemble"]
    for task in ["binary_classification", "regression"]:
        time_result = {
            "lte": [1],
            "mdi_default": [1],
            "mdi_tuned": [1],
            "shap_default": [1],
            "shap_tuned": [1],
            "pi_single": [1],
            "pi_ensemble": [1],
        }
        model_path_map = {
            "binary_classification": os.path.join(directory, 'models/LTE_models_clf'),
            "regression": os.path.join(directory, 'models/LTE_models_reg')
        }
        model_path = model_path_map[task]
        if task == "binary_classification":
            data_path = os.path.join(file_path, '[UCI]Covertype')
        elif task == "regression":
            data_path = os.path.join(file_path, '[kaggle]uber_and_lyft')
        train_x_ori, train_y_ori, valid_x_ori, valid_y_ori = multiple_data_to_million(data_path)

        for running_round in range(1, rounds + 1):
            logger.log("Start running at Round %d..." % running_round)
            train_x = pd.concat([train_x_ori] * running_round).reset_index(drop=True)
            train_y = pd.concat([train_y_ori] * running_round).reset_index(drop=True)
            valid_x = pd.concat([valid_x_ori] * running_round).reset_index(drop=True)
            valid_y = pd.concat([valid_y_ori] * running_round).reset_index(drop=True)
            logger.log("Shape of training dataset: %d" % train_x.shape[0])


            for eval_type in eval_list:
                logger.log("eval_type--%s" % eval_type)

                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                logger.log("Start time：", now)
                t0 = time.time()
                if eval_type == "mdi_default":
                    fi = get_mdi_default_result(train_x, train_y, valid_x, valid_y, n_jobs, task)
                elif eval_type == "mdi_tuned":
                    fi = get_mdi_tuned_result(train_x, train_y, valid_x, valid_y, n_jobs, task)
                elif eval_type == "shap_default":
                    fi = get_shap_default_result(train_x, train_y, valid_x, valid_y, n_jobs, task)
                elif eval_type == "shap_tuned":
                    fi = get_shap_tuned_result(train_x, train_y, valid_x, valid_y, n_jobs, task)
                elif eval_type == "pi_single":
                    fi = get_pi_single_result(train_x, train_y, valid_x, valid_y, n_jobs, task)
                elif eval_type == "pi_ensemble":
                    fi = get_pi_ensemble_result(train_x, train_y, valid_x, valid_y, n_jobs, task)
                else:
                    meta_features = get_meta_features(train_x, train_y, valid_x, valid_y, logger, task, n_jobs)
                    lte_result = get_LTE_result(meta_features, model_path, logger, task)
                running_time = time.time() - t0
                time_result[eval_type].append(running_time)
                logger.log("Time Spent in eval_type %s: %s" % (eval_type, running_time))
        if task == "binary_classification":
            prefix = "clf"
        elif task == "regression":
            prefix = "reg"
        with open(os.path.join(directory, "test/eval_result_files/%s_running_time_result.json" % prefix), "w") as f:
            json.dump(time_result, f)

    clf_res_data_path = os.path.join(directory, "test/eval_result_files/clf_running_time_result.json")
    reg_res_data_path = os.path.join(directory, "test/eval_result_files/reg_running_time_result.json")
    assert os.path.exists(clf_res_data_path)
    assert os.path.exists(reg_res_data_path)
    plot_fig4(clf_res_data_path, os.path.join(directory, "test/eval_result_files/clf_running_time_result.pdf"), simple)
    plot_fig4(reg_res_data_path, os.path.join(directory, "test/eval_result_files/reg_running_time_result.pdf"), simple)


if __name__ == "__main__":
    logger = Logger("time_eval_LTE")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="directory of FeatureLTE", type=str, default="FeatureLTE")
    parser.add_argument("-r", "--rounds", help="data multiple rounds", type=int, default="1")
    parser.add_argument("-s", "--simple", help="run the fast test without PI & SHAP-tuned", action="store_true")
    parser.add_argument("-n", "--n_jobs", help="evaluation type", type=int, default="-1")
    args = parser.parse_args()
    rounds = args.rounds
    simple = args.simple
    directory = args.directory
    n_jobs = args.n_jobs
    run()