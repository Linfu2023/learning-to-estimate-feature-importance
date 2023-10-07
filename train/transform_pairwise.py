from concurrent.futures import ProcessPoolExecutor
import numpy as np
import math
import pandas as pd
import json


def _fetch_qid_data(y, qid, eval_at=None):
    qid_unique, qid2indices, qid_inverse_indices = np.unique(qid, return_index=True, return_inverse=True)
    # get item releveance for each query id
    qid2rel = [[] for _ in range(len(qid_unique))]
    for i, qid_unique_index in enumerate(qid_inverse_indices):
        qid2rel[qid_unique_index].append(y[i])
    # get dcg, idcg for each query id @eval_at
    if eval_at:
        qid2dcg = [_CalcDCG(qid2rel[i][:eval_at]) for i in range(len(qid_unique))]
        qid2idcg = [_CalcDCG(sorted(qid2rel[i], reverse=True)[:eval_at]) for i in range(len(qid_unique))]
    else:
        qid2dcg = [_CalcDCG(qid2rel[i]) for i in range(len(qid_unique))]
        qid2idcg = [_CalcDCG(sorted(qid2rel[i], reverse=True)) for i in range(len(qid_unique))]
    return qid2indices, qid2rel, qid2idcg, qid2dcg


def _CalcDCG(labels):
    sumdcg = 0.0
    for i in range(len(labels)):
        rel = labels[i]
        if rel != 0:
            sumdcg += (rel) / math.log2(i + 2)
    return sumdcg


def get_training_instance(y, qid, qid_unique_idx):
    qid2indices, qid2rel, qid2idcg, _ = _fetch_qid_data(y, qid)
    IDCG = 1.0 / qid2idcg[qid_unique_idx]
    rel_list = qid2rel[qid_unique_idx]
    qid_start_idx = qid2indices[qid_unique_idx]
    X1_index = []
    X2_index = []
    weight = []
    Y = []
    for pos_idx in range(len(rel_list)):
        for neg_idx in range(len(rel_list)):
            if rel_list[pos_idx] <= rel_list[neg_idx]:
                continue

            # calculate lambda
            pos_loginv = 1.0 / math.log2(pos_idx + 2)
            neg_loginv = 1.0 / math.log2(neg_idx + 2)
            pos_label = rel_list[pos_idx]
            neg_label = rel_list[neg_idx]
            original = (pos_label) * pos_loginv + (neg_label) * neg_loginv
            changed = (neg_label) * pos_loginv + (pos_label) * neg_loginv
            delta = (original - changed) * IDCG
            if delta < 0:
                delta = -delta
            # balanced class
            if 1 != (-1) ** (qid_unique_idx + pos_idx + neg_idx):
                X1_index.append(qid_start_idx + pos_idx)
                X2_index.append(qid_start_idx + neg_idx)
                weight.append(delta)
                Y.append(1)
            else:
                X1_index.append(qid_start_idx + neg_idx)
                X2_index.append(qid_start_idx + pos_idx)
                weight.append(delta)
                Y.append(0)
    return X1_index, X2_index, weight, Y


def transform_pairwise(file_list, X, y, qid):
    print("Start transforming pairwise datasets.")
    qid2indices, qid2rel, qid2idcg, _ = _fetch_qid_data(y, qid)
    X1 = dict()
    X2 = dict()
    weight = dict()
    Y = dict()
    results = []
    ex = ProcessPoolExecutor(20)
    for qid_unique_idx in range(len(qid2indices)):
        file = file_list[qid_unique_idx]
        if qid2idcg[qid_unique_idx] == 0:
            continue
        result = [file, ex.submit(get_training_instance, y, qid, qid_unique_idx)]
        results.append(result)
    ex.shutdown(wait=True)
    print("Finish transforming.")
    for res in results:
        file = res[0]
        res = res[1]
        res = res.result()
        X1_index = res[0]
        X2_index = res[1]
        weight_file = list(res[2])
        Y_file = list(res[3])
        X1_file = []
        X2_file = []
        for idx1, idx2 in zip(X1_index, X2_index):
            X1_file.append(list(X[idx1]))
            X2_file.append(list(X[idx2]))
        X1[file] = X1_file
        X2[file] = X2_file
        weight[file] = weight_file
        Y[file] = Y_file
    print("Finish appending.")
    return X1, X2, Y, weight


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


if __name__ == '__main__':
    data = pd.read_csv('./save/meta_features_LTE_data_train.csv')
    label = pd.read_csv('./save/meta_features_LTE_labels_train_v1.csv')
    label['label'][label['label'] < 0.0005] = 0
    group_list = get_group_list(data)
    file_list = list(set(data['filename']))
    del data['filename']
    qid = []
    qid_count = 1
    for group in group_list:
        for j in range(group):
            qid.append(qid_count)
        qid_count += 1
    qid = np.array(qid)
    x1, x2, y, weight = transform_pairwise(file_list, data.values.astype(float), label.values.ravel(), qid)
    with open('./save/transformed_meta_features_data_06_v3_x1.json', 'w') as f:
        json.dump(x1, f)
    with open('./save/transformed_meta_features_data_06_v3_x2.json', 'w') as f:
        json.dump(x2, f)
    with open('./save/transformed_meta_features_data_06_v3_y.json', 'w') as f:
        json.dump(y, f)
    with open('./save/transformed_meta_features_data_06_v3_weight.json', 'w') as f:
        json.dump(weight, f)
