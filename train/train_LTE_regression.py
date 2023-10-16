import pandas as pd
import numpy as np
from logger import Logger
from sklearn.preprocessing import StandardScaler
from LambdaRank import LambdaRankNN
import tensorflow as tf
from transform_pairwise import transform_pairwise
from keras import initializers
import keras
import random
import pickle
import os
import copy
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
assert tf.__version__ == '2.3.1'
assert keras.__version__ == '2.4.3'

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                        inter_op_parallelism_threads=1,
                                        allow_soft_placement=True,
                                        device_count={"CPU": 1})

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


def process_cat(data, categorical_features):
    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes
        data[feature] = data[feature].astype('category')
    return data


def get_group_list(data):
    count = 0
    group_list = []
    file_now = data['filename'].iloc[0]
    for i in range(len(data)):
        if data['filename'].iloc[i] == file_now:
            count += 1
        else:
            group_list.append(count)
            file_now = data['filename'].iloc[i]
            count = 1
    group_list.append(count)
    return group_list


def get_qid(data):
    qid = []
    qid_count = 1
    group_list = get_group_list(data)
    for group in group_list:
        for j in range(group):
            qid.append(qid_count)
        qid_count += 1
    qid = np.array(qid)
    return qid


def get_data(data_train, label_train):
    qid = get_qid(data_train)

    file_list = list(data_train['filename'].unique())
    del data_train['filename']
    x1, x2, y, weight = transform_pairwise(file_list, data_train.values.astype(float),
                                           label_train.values.ravel(), qid)

    X1_trans = []
    X2_trans = []
    Y = []
    W = []
    for key in x1:
        X1_trans.extend(x1[key])
        X2_trans.extend(x2[key])
        Y.extend(y[key])
        W.extend(weight[key])
    X1_trans, X2_trans, Y, W = np.array(X1_trans), np.array(X2_trans), np.array(Y), np.array(W)
    return X1_trans, X2_trans, Y, W


def transform_results(pred):
    sorted_pred = copy.deepcopy(pred)
    temp = np.argsort(pred)
    for i in range(len(pred)):
        sorted_pred[temp[i]] = i
    sorted_pred = sorted_pred / (len(pred) - 1)
    return sorted_pred


def get_label(df):
    fname = df['filename'].unique()
    for f in fname:
        data = df[df['filename'] == f]
        m = np.unique(np.nanquantile(data['label'].values, [i / 5 for i in range(1, 5)]))
        ind = data[data['label'] > m[-1]].index.to_list()
        df['label'][ind] = 4
        for j in range(1, len(m)):
            ind = data[(data['label'] > m[j - 1]) & (data['label'] <= m[j])].index.to_list()
            df['label'][ind] = j
        ind = data[data['label'] <= m[0]].index.to_list()
        df['label'][ind] = 0
    return df['label']


def preprocess_data(data):
    data['label'][data['label'] < 0.0005] = 0
    data['label'][data['label'] > 0.1] = 0.1

    label = pd.DataFrame(data['label'])

    del data['label']

    return data, label


def train_and_predict(data_train, data_val, label_version, seed):
    try:
        logger.log("Finish ensembling data.")
        categorical_features = list(data_train.select_dtypes(exclude=np.number).columns)[1:]
        if 'filename' in categorical_features:
            raise NotImplementedError

        logger.log("Shape of input training meta data:")
        logger.log(data_train.shape)
        logger.log("Shape of input validation meta data:")
        logger.log(data_val.shape)

        features = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8']
        for i, feature in enumerate(features):
            SS = StandardScaler().fit(data_train[feature].values.reshape(-1, 1))
            data_train[feature] = SS.transform(data_train[feature].values.reshape(-1, 1))
            data_val[feature] = SS.transform(data_val[feature].values.reshape(-1, 1))

        data_train, label_train = preprocess_data(data_train)
        data_val, label_val = preprocess_data(data_val)

        X1_trans, X2_trans, Y, W = get_data(data_train, label_train)
        X1_trans_val, X2_trans_val, Y_val, W_val = get_data(data_val, label_val)
        d_train = {'X1': X1_trans, 'X2': X2_trans, 'Y': Y, 'W': W}
        d_val = {'X1': X1_trans_val, 'X2': X2_trans_val, 'Y': Y_val, 'W': W_val}
        with open(os.path.join(directory, f"data/train_data/d_train_{label_version}_reg.pkl"), "wb") as f:
            pickle.dump(d_train, f)
        with open(os.path.join(directory, f"data/valid_data/d_val_{label_version}_reg.pkl"), "wb") as f:
            pickle.dump(d_val, f)

        d_train = pickle.load(open(os.path.join(directory, f"data/train_data/d_train_{label_version}_reg.pkl"), 'rb'))
        d_val = pickle.load(open(os.path.join(directory, f"data/valid_data/d_val_{label_version}_reg.pkl"), 'rb'))
        X1_trans, X2_trans, Y, W = d_train['X1'], d_train['X2'], d_train['Y'], d_train['W']
        X1_trans_val, X2_trans_val, Y_val, W_val = d_val['X1'], d_val['X2'], d_val['Y'], d_val['W']

        logger.log(X1_trans.shape)
        logger.log(X1_trans_val.shape)

        ranker = LambdaRankNN()
        initializer = initializers.he_normal(seed=seed)

        ranker.build_model(input_shape=X1_trans.shape[1],
                           lr=0.0001733877252647961, weight_decay=0.0003602698769667812, solver='adam',
                           initializer=initializer, d_embedding=30,
                           n=47, sigma=0.25599866560131745, hidden=47)
        ranker.fit(X1_trans, X2_trans, Y, W, epochs=1000,
                   batch_size=1024, patience=10,
                   validation_data=([X1_trans_val, X2_trans_val], Y_val, W_val),
                   val_data_for_ndcg=[data_val, label_val])
        logger.log("finish fitting.")
        model_save_dir = os.path.join(directory, output_dir)
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        ranker.save(os.path.join(model_save_dir, 'LTE_%d_%d' % (seed, label_version)))

    except Exception:
        import traceback
        logger.log(traceback.format_exc())


def run(label_version, seed):
    try:
        # All the random seeds remain consistent.
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

        data_train = pd.read_csv(os.path.join(directory, 'data/train_data/meta_features_LTE_train_v%d_reg.csv' % label_version))
        data_val = pd.read_csv(os.path.join(directory, 'data/valid_data/meta_features_LTE_valid_v%d_reg.csv' % label_version))

        data_train = data_train.drop(['feature'], axis=1).fillna(0)
        data_val = data_val.drop(['feature'], axis=1).fillna(0)

        train_and_predict(data_train, data_val, label_version, seed)
    except Exception:
        import traceback
        logger.log(traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="directory of FeatureLTE", type=str, default="FeatureLTE")
    parser.add_argument("-o", "--output_dir", help="output directory of models", type=str, default="models/temp_LTE_models_reg")
    args = parser.parse_args()
    directory = args.directory
    output_dir = args.output_dir

    from concurrent.futures import ProcessPoolExecutor

    logger = Logger('train_LambdaRank_LTE')

    ex = ProcessPoolExecutor(5)
    try:
        for label_version in [1, 2, 3, 4, 5]:
            for seed in [1, 2, 3, 4, 5]:
                logger.log([label_version, seed])
                ex.submit(run, label_version, seed)
        ex.shutdown(wait=True)

    except Exception:
        import traceback

        logger.log(traceback.format_exc())
