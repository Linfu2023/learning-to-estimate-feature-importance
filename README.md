# FeatureLTE

FeatureLTE: Learning to Estimate Feature Importance

FeatureLTE is a learning-based technique of estimating feature importance. It learns pre-computed high-quality feature
importance scores from a large number of datasets. We build our pre-trained models for binary classification and regression problems using observations from nearly
1,000 public datasets, and the models can be used to predict feature importance score of the input dataset.

## Pre-trained LTE models

We provide the pre-trained LTE models in the repository, see:

* [`models/LTE_models_clf`](models/LTE_models_clf) for binary classification tasks with 25 different models in total
* [`models/LTE_models_reg`](models/LTE_models_reg)  for regression tasks with 25 different models in total

The specific model name e.g. `LTE_s1_v3` stands for:
* set random_seed=3 when generate meta datasets for training 
* set random_seed=1 when training LTE models

## Public datasets
We build our pre-trained models for binary classification and regression problems using observations from nearly **1,000 public datasets**.
Dur to the space limitations of the repository, we only provide the test datasets, see [`data/test_data`](data/test_data).

We list all the datasets used for FeatureLTE at [`data/public_datasets_list`](data/public_datasets_list).  They can be found at public websites directly. 
* [`train_datasets_clf.txt`](data/public_datasets_list/train_datasets_clf.txt): training datasets used in binary classification FeatureLTE  
* [`train_datasets_reg.txt`](data/public_datasets_list/train_datasets_reg.txt): training datasets used in regression FeatureLTE
* [`valid_datasets_clf.txt`](data/public_datasets_list/valid_datasets_clf.txt): validation datasets used in binary classification FeatureLTE
* [`valid_datasets_reg.txt`](data/public_datasets_list/valid_datasets_reg.txt): validation datasets used in regression FeatureLTE
* [`test_datasets_clf.txt`](data/public_datasets_list/test_datasets_clf.txt): test datasets used in binary classification FeatureLTE
* [`test_datasets_reg.txt`](data/public_datasets_list/test_datasets_reg.txt): test datasets used in regression FeatureLTE

The dataset provided in the list above is in the format of [data_source]data_name, e.g., [openml]BNG(lymph,nominal,1000000) refers this dataset at https://www.openml.org/search?type=data&sort=runs&id=76&status=active

Since our model was trained using a **meta-learning** approach, we have placed the meta datasets used for training, validation and test in the repository. See the folder [`data/train_data`](data/train_data) & [`data/valid_data`](data/valid_data) & [`data/test_data`](data/test_data).  
[**Training FeatureLTE Model from Scratch**](#Training FeatureLTE Model from Scratch)  will introduce how to reproduce the LTE models with the meta datasets.

## Installing Dependencies

We build the FeatureLTE project under `python=3.6.5`, prepare the python environment of FeatureLTE by executing the scripts below:


```bash
$ cd FeatureLTE
$ pip install -r requirements.txt
```

## Training FeatureLTE Model from Scratch

You can also train an LTE model from scratch, simply run the code:

* binary classification task:

```bash
$ cd FeatureLTE/scripts
$ sh run_train_LTE_classification.sh
```

Then it will start to run the training scripts of reproducing the LTE models of binary classification tasks.
The models will be saved into the path from argument `output_dir`, default is[`models/LTE_models_clf`](models/LTE_models_clf). Similarly, the LTE models of regression tasks can be reproduced by running:

* regression task:

```bash
$ cd FeatureLTE/scripts
$ sh run_train_LTE_regression.sh
```
The models will be saved into the path from argument `output_dir`, default is [`models/LTE_models_reg`](models/LTE_models_reg). 

## Predict with LTE models

You can run a quick prediction demo by executing the scripts below:
```bash
$ cd FeatureLTE/scripts
$ sh run_predict.sh
```
This script can generate the *Feature Importance Score* by providing the arguments:
* `directory` root directory of FeatureLTE
* `file_name` specify a certain dataset from folder [`data/test_data`](data/test_data), default is "[UCI]Arrhythmia"
* `task` specify a binary classification task or a regression task, default is "binary_classification"
* `model_dir` choose the models in folder [`models`](models), depends on a binary classification or regression task, default is "LTE_models_clf"


Every test dataset has been split into 5 folds randomly: `<file_name>_eval_0` ~ `<file_name>_eval_4`, the scripts will proceed with
all these 5 datasets in parallel,
It saves the feature importance scores into CSV file `lte_FI_result.csv`, at the directory of each dataset. 

Here is one of the LTE prediction results that will be saved by executing the scripts above with default arguments:

 `data/test_data/binary_classification/[UCI]Arrhythmia/[UCI]Arrhythmia_eval_0/lte_FI_result.csv`.  


| feature_name | V1     | V2     | V3     | V4     | V5     | mean   |
|--------------|--------|--------|--------|--------|--------|--------|
| f0           | 0.9386 | 0.9386 | 0.9233 | 0.9386 | 0.9386 | 0.9356 |
| f1           | 0.8850 | 0.8352 | 0.8620 | 0.8544 | 0.8659 | 0.8605 |
| f2           | 0.6667 | 0.6206 | 0.6245 | 0.6283 | 0.6091 | 0.6298 |
| f3           | 0.7241 | 0.7203 | 0.7011 | 0.7279 | 0.7049 | 0.7157 |
| ...          | ...    | ...    | ...    | ...    | ...    |  ...   |

* **V1~V5** indicate the model version, which means the FIS is predicted by the LTE model in seed=1, **mean** is calculated as the average of these five scores
* The LTE scores are positively correlated with the ranks of the features (a higher rank corresponds to a higher LTE score)

Also the *meta features* generated during the prediction will be saved into json file `meta_features_LTE.json`, at the
directory of each dataset.   
e.g.,  `data/test_data/binary_classification/[UCI]Arrhythmia/[UCI]Arrhythmia_eval_4/meta_features_LTE.json`.  

## Evaluate an LTE Model

After the LTE model was successfully trained and saved, we can evaluate the model by the output of the test dataset:
First make sure you have performed the previous step [**Predict with LTE models**](#Predict with LTE models) and produced feature importance scores
into `lte_FI_result.csv`.

Then you can run a quick evaluation demo by executing the scripts below:

```bash
$ cd FeatureLTE/scripts
$ sh run_eval.sh
```

This script can generate the *AUC Score* by modify the arguments:

* `directory` root directory of FeatureLTE
* `file_name` specify a certain dataset from folder [`data/test_data`](data/test_data), default is "[UCI]Arrhythmia"
* `eval_type` choose the FIS method, with options:
    * lte (default)
    * mdi_default
    * mdi_tuned
    * shap_default
    * shap_tuned
    * pi_single
    * pi_ensemble
* `task` specify a binary classification task or a regression task, default is "binary_classification"

At the evaluation phase, we will train fine-tuned boosted tree models using the test data with complete feature set and
feature subsets, which was selected based on the prediction result (top 5%, 10%, 15%, 20%).  
The evaluation result will be saved into json file `<eval_type>_eval_result.json`, at the directory of each dataset.

Here is one of the LTE evaluation results that will be saved by executing the scripts above with default arguments:
`data/test_data/binary_classification/[UCI]Arrhythmia/[UCI]Arrhythmia_eval_0/lte_eval_result.json`.   
```json
{
  "seed1": {
    "k5": 0.8635416666666667,
    "k10": 0.9015625,
    "k15": 0.8841145833333334,
    "k20": 0.8619791666666667,
    "k100": 0.8895833333333333
  },
  "seed2": {
    "k5": 0.8635416666666667,
    "k10": 0.8807291666666667,
    "k15": 0.8664062499999999,
    "k20": 0.8619791666666667,
    "k100": 0.8895833333333333
  },
  "seed3": {
    "k5": 0.8635416666666667,
    "k10": 0.8791666666666667,
    "k15": 0.8664062499999999,
    "k20": 0.8619791666666667,
    "k100": 0.8895833333333333
  },
  "seed4": {
    "k5": 0.8572916666666666,
    "k10": 0.8619791666666666,
    "k15": 0.8664062499999999,
    "k20": 0.8619791666666667,
    "k100": 0.8895833333333333
  },
  "seed5": {
    "k5": 0.875,
    "k10": 0.8807291666666667,
    "k15": 0.8515625,
    "k20": 0.8619791666666667,
    "k100": 0.8895833333333333
  }
}
```

* The key **seed1** indicates the model version, which means the result is evaluated under the FIS that predicted by the
  LTE model in seed=1
* **k5** means use the top 5% feature subset of the FIS result to train the fine-tuned LightGBM model (**k100** means use
  the complete feature set)
* The values are 
  * **AUC Scores** from binary classification tasks
  * **MAPE Scores** from regression tasks

## Reproduce the Evaluation Result
We have evaluated our LTE models on the test dataset from folder [`data/test_data`](data/test_data), the evaluation result can be reproduced by directly executing the scripts below:
```bash
$ cd FeatureLTE/scripts
$ sh run_reproduce.sh
```
It will evaluate on all the test datasets with different evaluation method, the plot result will be saved into  
* [`test/eval_result_files/clf_eval_result_fig2.pdf`](test/eval_result_files/clf_eval_result_fig2.pdf)
* [`test/eval_result_files/reg_eval_result_fig2.pdf`](test/eval_result_files/reg_eval_result_fig2.pdf)

**It is noteworthy that our previous experiments were conducted within the company's large-scale distributed system, whereas this script runs in a serial manner, which will take a considerable amount of time to complete.**

Those plots are consistent with those presented in Figure 2 of our paper.

## Evaluate the Running time
We discussed the efficiency of FIS estimation for LTE models in our paper, and the running time of different methods are showed in Figure 4. You can run a quick evaluation demo by directly executing the scripts below:
```bash
$ cd FeatureLTE/scripts
$ sh run_eval_running_time.sh
```
The test datasets are:
* binary classification: [`data/test_data/running_time_test/[UCI]Covertype`](data/test_data/running_time_test/[UCI]Covertype)
* regression: [`data/test_data/running_time_test/uber_and_lyft`](data/test_data/running_time_test/uber_and_lyft)  

It will evaluate on both of them, you can modify the argument in [`scripts/run_eval_running_time.sh`](scripts/run_eval_running_time.sh) to control the datasize and specify methods:

The arguments are:
* `rounds`: we evaluate from datasize: 1 million to 7 million, you can set this argument from 1 to 7 to adjust, default is 1
* `simple`: since method like PI-ensemble takes a really long time to evaluate (over 100,000 seconds on 1 million scale), the scripts default runs on LTE, MDI-default, MDI-tuned and SHAP-default, which can be done in 1000 seconds on 1 million data size.

If you want to reproduce the full results, please set the following parameters:
```bash
python ../test/eval_running_time.py --rounds 7 --directory ../ --n_jobs 8
```

and the specific numbers will be saved into 
* `test/eval_result_files/clf_running time_result.json`
* `test/eval_result_files/reg_running time_result.json`


The plot result will be saved into 
* `test/eval_result_files/clf_running time_result.pdf`
* `test/eval_result_files/reg_running time_result.pdf`


Here is one of the running time evaluation results that will be saved by executing the scripts above with default arguments:  
`clf_running time_result.json`

**It's worth noting that our results were obtained on a MacBook with 8 cores and 16GB memory, and different devices may yield varying numerical results.**


```json
{
    "lte":
    [1, 164.17056226730347],
    "mdi_default":
    [1, 5.932538032531738],
    "mdi_tuned":
    [1, 219.6426095962524],
    "shap_default":
    [1, 52.58369016647339],
    "shap_tuned":
    [1],
    "pi_single":
    [1],
    "pi_ensemble":
    [1]
}
```
Running time of each method will be list in the file:  
e.g., `"lte":[1, 164.17056226730347]` means under method LTE, the running time consumed is 164 seconds with 1 million datasize, the first element of the list is always `1`, which is an initial value  merely to ensure that the starting point is 0 after log calculation when plotting.
If you have modified the argument `rounds > 1`, then more values will be appended to the list.