import os
import numpy as np
import json
import pandas as pd

KEY_MAPPER = {
    "mdi_default": "MDI-default",
    "mdi_tuned": "MDI-tuned",
    "shap_default": "SHAP-default",
    "shap_tuned": "SHAP-tuned",
    "pi_single": "PI-single",
    "pi_ensemble": "PI-ensemble",
    "lte": "LTE",
}

CD_DIAGRAM_COLS = ['classifier_name', 'dataset_name', 'k5', 'k10', 'k15', 'k20']

def merge_eval_results(test_data_list, directory):
    eval_result_clf = {
        "lte": {},
        "mdi_default": {},
        "mdi_tuned": {},
        "shap_default": {},
        "shap_tuned": {},
        "pi_single": {},
        "pi_ensemble": {},
    }

    eval_result_reg = {
        "lte": {},
        "mdi_default": {},
        "mdi_tuned": {},
        "shap_default": {},
        "shap_tuned": {},
        "pi_single": {},
        "pi_ensemble": {},
    }

    for file_name, task in test_data_list:
        tmp_res_dict = {
            "lte": [[], [], [], []],
            "mdi_default": [[], [], [], []],
            "mdi_tuned": [[], [], [], []],
            "shap_default": [[], [], [], []],
            "shap_tuned": [[], [], [], []],
            "pi_single": [[], [], [], []],
            "pi_ensemble": [[], [], [], []],
        }
        file_path = os.path.join(directory, 'data/test_data/%s/' % task + file_name)
        for rank in range(5):
            eval_file_path = os.path.join(file_path, file_name + '_eval_%d' % rank)
            for eval_type in ["lte", "mdi_default", "mdi_tuned", "shap_default", "shap_tuned", "pi_single",
                              "pi_ensemble"]:
                with open(os.path.join(eval_file_path, "%s_eval_result.json" % eval_type), 'r') as f:
                    tmp_res = json.load(f)
                if eval_type == "lte":
                    for seed in range(1, 6):
                        k100 = tmp_res['seed%d' % seed]['k100']
                        tmp_res_dict['lte'][0].append(k100 - tmp_res['seed%d' % seed]['k5'])
                        tmp_res_dict['lte'][1].append(k100 - tmp_res['seed%d' % seed]['k10'])
                        tmp_res_dict['lte'][2].append(k100 - tmp_res['seed%d' % seed]['k15'])
                        tmp_res_dict['lte'][3].append(k100 - tmp_res['seed%d' % seed]['k20'])
                else:
                    k100 = tmp_res['k100']
                    tmp_res_dict[eval_type][0].append(k100 - tmp_res['k5'])
                    tmp_res_dict[eval_type][1].append(k100 - tmp_res['k10'])
                    tmp_res_dict[eval_type][2].append(k100 - tmp_res['k15'])
                    tmp_res_dict[eval_type][3].append(k100 - tmp_res['k20'])

        if task == "binary_classification":
            for eval_type in ["lte", "mdi_default", "mdi_tuned", "shap_default", "shap_tuned", "pi_single", "pi_ensemble"]:
                eval_result_clf[eval_type][file_name] = [
                    np.mean(tmp_res_dict[eval_type][0]),
                    np.mean(tmp_res_dict[eval_type][1]),
                    np.mean(tmp_res_dict[eval_type][2]),
                    np.mean(tmp_res_dict[eval_type][3]),
                ]
        else:
            for eval_type in ["lte", "mdi_default", "mdi_tuned", "shap_default", "shap_tuned", "pi_single", "pi_ensemble"]:
                eval_result_reg[eval_type][file_name] = [
                    np.mean(tmp_res_dict[eval_type][0]),
                    np.mean(tmp_res_dict[eval_type][1]),
                    np.mean(tmp_res_dict[eval_type][2]),
                    np.mean(tmp_res_dict[eval_type][3]),
                ]

    mean_var_dict_clf = {}
    mean_var_dict_reg = {}
    cd_diagram_clf = []
    cd_diagram_reg = []
    for eval_type in ["lte", "mdi_default", "mdi_tuned", "shap_default", "shap_tuned", "pi_single", "pi_ensemble"]:
        k5, k10, k15, k20 = [], [], [], []
        for k, v in eval_result_clf[eval_type].items():
            k5.append(v[0])
            k10.append(v[1])
            k15.append(v[2])
            k20.append(v[3])
            cd_diagram_clf.append([KEY_MAPPER[eval_type], k, v[0], v[1], v[2], v[3]])
        mean5, var5 = np.mean(k5), np.var(k5)
        mean10, var10 = np.mean(k10), np.var(k10)
        mean15, var15 = np.mean(k15), np.var(k15)
        mean20, var20 = np.mean(k20), np.var(k20)
        mean_var_dict_clf[eval_type] = [mean5, mean10, mean15, mean20, var5, var10, var15, var20]

        k5, k10, k15, k20 = [], [], [], []
        for k, v in eval_result_reg[eval_type].items():
            k5.append(v[0])
            k10.append(v[1])
            k15.append(v[2])
            k20.append(v[3])
            cd_diagram_reg.append([KEY_MAPPER[eval_type], k, v[0], v[1], v[2], v[3]])
        mean5, var5 = np.mean(k5), np.var(k5)
        mean10, var10 = np.mean(k10), np.var(k10)
        mean15, var15 = np.mean(k15), np.var(k15)
        mean20, var20 = np.mean(k20), np.var(k20)
        mean_var_dict_reg[eval_type] = [mean5, mean10, mean15, mean20, var5, var10, var15, var20]


    with open(os.path.join(directory, "test/eval_result_files/clf_eval_result_fig2.txt"), "w") as f1:
        for k in ["mdi_default", "mdi_tuned", "shap_default", "shap_tuned", "pi_single", "pi_ensemble", "lte"]:
            content = KEY_MAPPER[k] + ';' + ','.join(str(i) for i in mean_var_dict_clf[k][:4]) + ';' + ','.join(str(i) for i in mean_var_dict_clf[k][4:]) + '\n'
            f1.write(content)

    with open(os.path.join(directory, "test/eval_result_files/reg_eval_result_fig2.txt"), "w") as f2:
        for k in ["mdi_default", "mdi_tuned", "shap_default", "shap_tuned", "pi_single", "pi_ensemble", "lte"]:
            content = KEY_MAPPER[k] + ';' + ','.join(str(i) for i in mean_var_dict_reg[k][:4]) + ';' + ','.join(str(i) for i in mean_var_dict_reg[k][4:]) + '\n'
            f2.write(content)

    cd_diagram_clf = sorted(cd_diagram_clf, key=lambda x: (x[0], x[1]))
    cd_diagram_reg = sorted(cd_diagram_reg, key=lambda x: (x[0], x[1]))

    cd_diagram_clf_df = pd.DataFrame(cd_diagram_clf, columns=CD_DIAGRAM_COLS)
    cd_diagram_reg_df = pd.DataFrame(cd_diagram_reg, columns=CD_DIAGRAM_COLS)

    cd_diagram_clf_df.to_csv(os.path.join(directory, "test/eval_result_files/clf_eval_result_fig2.csv"), index=False)
    cd_diagram_reg_df.to_csv(os.path.join(directory, "test/eval_result_files/reg_eval_result_fig2.csv"), index=False)

