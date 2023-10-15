import argparse
import gzip
import os
import json
from sklearn.metrics import roc_auc_score

from train.logger import Logger
from eval_util import *


def process_cat_fea(data, categorical_features):
    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes
        data[feature] = data[feature].astype('category')
    return data


def evaluation(train_x, train_y, valid_x, valid_y, test_x, test_y, fi, n_jobs):
    seed_result = dict()
    for k in [5, 10, 15, 20]:
        logger.log("Start running file %s in k%% %s" % (file_name, str(k)))
        train_x_sub, valid_x_sub, test_x_sub = selection_according_to_prediction(train_x, valid_x, test_x, fi,
                                                                                 percent=0.01 * k)

        best_params, cv_auc = gridsearch_tuning(train_x_sub, train_y, valid_x_sub, valid_y, n_jobs)
        logger.log("============best params===================")
        logger.log(best_params)

        gbm = lgb.LGBMClassifier(**best_params)

        gbm.fit(train_x_sub.values, train_y.values.ravel())
        pred = gbm.predict_proba(test_x_sub)
        pred = pred[:, 1]
        seed_result[" k%s" % str(k)] = roc_auc_score(test_y.values.ravel(), pred)
        logger.log("Finish running file %s in  k %s" % (file_name, str(k)))
    return seed_result


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


def run():
    logger.log("Start running file %s" % file_name)
    file_path = os.path.join(directory, 'data/test_data/' + file_name)
    eval_file_path = os.path.join(file_path, file_name + '_eval_%d' % rank)

    logger.log("Loading data, please wait.")
    load_data(eval_file_path)

    train_x = pd.read_csv(os.path.join(eval_file_path, 'train_x.csv'))
    train_y = pd.read_csv(os.path.join(eval_file_path, 'train_y.csv'))
    valid_x = pd.read_csv(os.path.join(eval_file_path, 'valid_x.csv'))
    valid_y = pd.read_csv(os.path.join(eval_file_path, 'valid_y.csv'))
    test_x = pd.read_csv(os.path.join(eval_file_path, 'test_x.csv'))
    test_y = pd.read_csv(os.path.join(eval_file_path, 'test_y.csv'))
    categorical_features = list(train_x.select_dtypes(exclude=np.number).columns)
    train_x = process_cat_fea(train_x, categorical_features)
    valid_x = process_cat_fea(valid_x, categorical_features)
    test_x = process_cat_fea(test_x, categorical_features)
    logger.log("Successfully load the data.")
    eval_result = dict()
    if eval_type == "lte":
        local_fi_path = os.path.join(eval_file_path, "lte_result.csv")
        assert os.path.exists(local_fi_path)

        lte_result = pd.read_csv(local_fi_path, index=False)
        for seed in range(1, 6):
            logger.log("Start running file %s in seed %d" % (file_name, seed))
            fi = dict(zip(lte_result['feature_name'], lte_result['v%d' % seed]))
            seed_result = evaluation(train_x, train_y, valid_x, valid_y, test_x, test_y, fi, n_jobs)
            eval_result["seed%d" % seed] = seed_result
    else:
        if eval_type == "mdi-default":
            fi = get_mdi_default_result(train_x, train_y, valid_x, valid_y)
        elif eval_type == "mdi-tuned":
            fi = get_mdi_tuned_result(train_x, train_y, valid_x, valid_y)
        elif eval_type == "shap-default":
            fi = get_shap_default_result(train_x, train_y, valid_x, valid_y)
        elif eval_type == "shap-tuned":
            fi = get_shap_tuned_result(train_x, train_y, valid_x, valid_y)
        elif eval_type == "pi-single":
            fi = get_pi_single_result(train_x, train_y, valid_x, valid_y)
        elif eval_type == "=pi-ensemble":
            fi = get_pi_ensemble_result(train_x, train_y, valid_x, valid_y)
        else:
            raise ValueError("Invalid eval type: %s!" % eval_type)
        with open(os.path.join(eval_file_path, "%s_result.json" % eval_type), 'w') as f:
            json.dump(fi, f)
        eval_result = evaluation(train_x, train_y, valid_x, valid_y, test_x, test_y, fi, n_jobs)
    return eval_result


if __name__ == "__main__":
    logger = Logger("evaluation_LTE")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="directory of FeatureLTE", type=str, default="FeatureLTE")
    parser.add_argument("-f", "--file_name", help="file name of test data", type=str, default="[UCI]Arrhythmia")
    parser.add_argument("-e", "--eval_type", help="evaluation type", type=str, default="lte")
    parser.add_argument("-n", "--n_jobs", help="evaluation type", type=str, default="lte")
    args = parser.parse_args()
    file_name = args.file_name
    directory = args.directory
    eval_type = args.eval_type
    n_jobs = args.n_jobs
    from concurrent.futures import ProcessPoolExecutor

    ex = ProcessPoolExecutor(5)
    try:
        for rank in range(5):
            logger.log("rank: %d" % rank)
            ex.submit(run)
        ex.shutdown(wait=True)
    except Exception:
        import traceback

        logger.log(traceback.format_exc())
