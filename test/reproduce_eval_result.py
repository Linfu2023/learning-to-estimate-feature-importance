import argparse
import os
import sys
import json
import gzip

sys.path.append("..")
from train.logger import Logger
from predict.generate_meta_features_clf import get_meta_features_from_csv_clf
from predict.generate_meta_features_reg import get_meta_features_from_csv_reg
from predict.predict import get_LTE_result
from eval_utils import *
from eval_on_test_data import evaluation
from data_utils import merge_eval_results
from plot_utils import plot_fig2, plot_fig3


def get_test_data_list():
    list_file_path = os.path.join(directory, 'data/public_datasets_list')
    with open(os.path.join(list_file_path, 'test_datasets_clf'), 'r') as f1:
        test_list_clf = f1.readlines()
    with open(os.path.join(list_file_path, 'test_datasets_reg'), 'r') as f2:
        test_list_reg = f2.readlines()

    test_data_list = [(i.strip(), "binary_classification") for i in test_list_clf] +\
                     [(j.strip(), "regression") for j in test_list_reg]

    return test_data_list


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

    for csv_name in ["train_x", "train_y", "valid_x", "valid_y", "test_x", "test_y"]:
        output_file = os.path.join(eval_file_path, "temp.gz")
        merge_gzip_files(output_file)

        with gzip.open(output_file, 'rb') as f_in:
            with open(os.path.join(eval_file_path, f'{csv_name}.csv'), 'wb') as f_out:
                for line in f_in:
                    f_out.write(line)
        os.remove(output_file)


def get_test_data(eval_file_path):
    logger.log("Loading data, please wait.")
    load_data(eval_file_path)
    # load data
    train_x = pd.read_csv(os.path.join(eval_file_path, 'train_x.csv'))
    train_y = pd.read_csv(os.path.join(eval_file_path, 'train_y.csv'))
    valid_x = pd.read_csv(os.path.join(eval_file_path, 'valid_x.csv'))
    valid_y = pd.read_csv(os.path.join(eval_file_path, 'valid_y.csv'))
    test_x = pd.read_csv(os.path.join(eval_file_path, 'test_x.csv'))
    test_y = pd.read_csv(os.path.join(eval_file_path, 'test_y.csv'))
    categorical_features = list(train_x.select_dtypes(exclude=np.number).columns)
    train_x = process_cat(train_x, categorical_features)
    valid_x = process_cat(valid_x, categorical_features)
    test_x = process_cat(test_x, categorical_features)

    logger.log("Successfully load the data.")

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_meta_features(train_x, train_y, valid_x, valid_y, logger, task, n_jobs):
    data_x = pd.concat([train_x, valid_x])
    data_y = pd.concat([train_y, valid_y])
    categorical_features = list(data_x.select_dtypes(exclude=np.number).columns)
    numerical_features = list(data_x.select_dtypes(include=np.number).columns)
    if task == 'binary_classification':
        meta_features = get_meta_features_from_csv_clf(data_x, data_y, numerical_features, categorical_features, logger, n_jobs)
    elif task == 'regression':
        meta_features = get_meta_features_from_csv_reg(data_x, data_y, numerical_features, categorical_features, logger, n_jobs)

    return meta_features


def get_eval_result(eval_type, eval_file_path, train_x, train_y, valid_x, valid_y, test_x, test_y, logger, task):
    eval_result = dict()
    if eval_type == "lte":
        local_fi_path = os.path.join(eval_file_path, "lte_FI_result.csv")
        assert os.path.exists(local_fi_path)
        lte_result = pd.read_csv(local_fi_path)
        for seed in range(1, 6):
            logger.log("Start running file %s in seed %d" % (eval_file_path, seed))
            fi = dict(zip(lte_result['feature_name'], lte_result['V%d' % seed]))
            seed_result = evaluation(train_x, train_y, valid_x, valid_y, test_x, test_y, fi, n_jobs,
                                     logger, task)
            eval_result["seed%d" % seed] = seed_result
    else:
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
            raise ValueError("Invalid eval type: %s!" % eval_type)
        with open(os.path.join(eval_file_path, "%s_FI_result.json" % eval_type), 'w') as f:
            json.dump(fi, f)
        eval_result = evaluation(train_x, train_y, valid_x, valid_y, test_x, test_y, fi, n_jobs, logger, task)

    return eval_result


def run():
    # get the test datasets list
    test_data_list = get_test_data_list()
    model_path_map = {
        "binary_classification": os.path.join(directory, 'models/LTE_models_clf'),
        "regression": os.path.join(directory, 'models/LTE_models_reg')
    }
    # run the evaluation pipeline
    for file_name, task in test_data_list:
        file_path = os.path.join(directory, 'data/test_data/%s/' % task + file_name)
        model_path = model_path_map[task]
        for rank in range(5):
            logger.log("Current file name: %s, trial %d" % (file_name, rank))
            eval_file_path = os.path.join(file_path, file_name + '_eval_%d' % rank)
            train_x, train_y, valid_x, valid_y, test_x, test_y = get_test_data(eval_file_path)
            # get meta features
            logger.log("Start to get meta features...")
            meta_features = get_meta_features(train_x, train_y, valid_x, valid_y, logger, task, n_jobs)
            with open(os.path.join(eval_file_path, 'meta_features_LTE.json'), 'w') as f:
                json.dump(meta_features, f)
            logger.log("Successfully save the meta features into file %s" % os.path.join(eval_file_path, 'meta_features_LTE.json'))

            # get LTE result
            logger.log("Start to get LTE result...")
            lte_result = get_LTE_result(meta_features, model_path, logger, task)
            logger.log("Successfully get the LTE result of file %s in trial %d" % (file_name, rank))
            lte_result.to_csv(os.path.join(eval_file_path, "lte_FI_result.csv"), index=False)
            logger.log("Successfully save the LTE result into file %s" % os.path.join(eval_file_path, "lte_FI_result.csv"))

            # get evaluation result
            logger.log("Start to get evaluate result...")
            for eval_type in ["lte", "mdi_default", "mdi_tuned", "shap_default", "shap_tuned", "pi_single", "pi_ensemble"]:
                logger.log("eval in type: %s" % eval_type)
                eval_result = get_eval_result(eval_type, eval_file_path, train_x, train_y, valid_x, valid_y, test_x, test_y, logger, task)
                with open(os.path.join(eval_file_path, "%s_eval_result.json" % eval_type), 'w') as f:
                    json.dump(eval_result, f)
                logger.log("Successfully save the eval result into file %s" % os.path.join(eval_file_path, "%s_eval_result.json" % eval_type))

    # merge the eval results of fig2
    merge_eval_results(test_data_list, directory)
    clf_fig2_data_path = os.path.join(directory, "test/eval_result_files/clf_eval_result_fig2.txt")
    reg_fig2_data_path = os.path.join(directory, "test/eval_result_files/reg_eval_result_fig2.txt")
    clf_fig3_data_path = os.path.join(directory, "test/eval_result_files/clf_eval_result_fig3.csv")
    reg_fig3_data_path = os.path.join(directory, "test/eval_result_files/reg_eval_result_fig3.csv")
    assert os.path.exists(clf_fig2_data_path)
    assert os.path.exists(reg_fig2_data_path)
    assert os.path.exists(clf_fig3_data_path)
    assert os.path.exists(reg_fig3_data_path)
    plot_fig2(clf_fig2_data_path, os.path.join(directory, "test/eval_result_files/clf_eval_result_fig2.pdf"))
    plot_fig2(reg_fig2_data_path, os.path.join(directory, "test/eval_result_files/reg_eval_result_fig2.pdf"))
    plot_fig3(clf_fig3_data_path, os.path.join(directory, "test/eval_result_files/clf_eval_result_fig3.pdf"))
    plot_fig3(reg_fig3_data_path, os.path.join(directory, "test/eval_result_files/reg_eval_result_fig3.pdf"))


if __name__ == "__main__":
    logger = Logger("reproduce_evaluation_LTE")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="directory of FeatureLTE", type=str, default="FeatureLTE")
    parser.add_argument("-n", "--n_jobs", help="evaluation type", type=int, default="8")
    args = parser.parse_args()
    directory = args.directory
    n_jobs = args.n_jobs
    run()