from ast import arg
import os
from os.path import join as pjoin
import json
import jsonlines
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

LOCAL_SAVES_ROOT = "../LLaMA-Factory/saves/"
ONLINE_SAVES_ROOT = "../LLM-online/saves/"

LOCAL_PRED_PATTERN = LOCAL_SAVES_ROOT + "{}/{}/lora_sft_pred/generated_predictions.jsonl"
ONLINE_PRED_PATTERN = ONLINE_SAVES_ROOT + "{}/{}/generated_predictions.jsonl"

TEST_INFOS_FILE = "./test_infos.json"

EVAL_ALL_RESULT = "./results.json"
RQ1_TABLE = "./rq1_table.csv"
RQ3_TABLE = "./rq3_table.csv"
PRELIMINARY_TABLE = "./preliminary_table.csv"


def eval_single_results(results):
    for p in results:
        p['label'] = p['label'].strip().lower()
        p['predict'] = p['predict'].strip().lower()
    
    labels = [p['label'] == 'yes' for p in results]
    preds = ['yes' in p['predict'] for p in results]
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    tp, fp, tn, fn = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn + 1e-8)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'fpr': fpr
    }


def get_all_predictions():
    all_predictions = {}
    for dataset in os.listdir(LOCAL_SAVES_ROOT):
        if dataset not in all_predictions:
            all_predictions[dataset] = {}
        for model in os.listdir(pjoin(LOCAL_SAVES_ROOT, dataset)):
            print(f"Obtaining predictions for {dataset}/{model}")
            try:
                with jsonlines.open(LOCAL_PRED_PATTERN.format(dataset, model)) as reader:
                    all_predictions[dataset][model] = [p for p in reader]
            except FileNotFoundError:
                print(f"File not found for {dataset}/{model}")
                continue
            
    for dataset in os.listdir(ONLINE_SAVES_ROOT):
        if dataset not in all_predictions:
            all_predictions[dataset] = {}
        for model in os.listdir(pjoin(ONLINE_SAVES_ROOT, dataset)):
            print(f"Obtaining predictions for {dataset}/{model}")
            try:
                with jsonlines.open(ONLINE_PRED_PATTERN.format(dataset, model)) as reader:
                    all_predictions[dataset][model] = [p for p in reader]
            except FileNotFoundError:
                print(f"File not found for {dataset}/{model}")
                continue
    
    return all_predictions
    

# def eval_all(all_predictions):
#     results = {}
#     for dataset, models in all_predictions.items():
#         results[dataset] = {}
#         for model, predictions in models.items():
#             print(f"Evaluating {dataset}/{model}")
#             eval_result = eval_single_results(predictions)
#             results[dataset][model] = eval_result
            
#     with open("./results.json", "w") as f:
#         json.dump(results, f, indent=4)


def divide_by_optim_level(predictions):
    with open(TEST_INFOS_FILE, "r") as f:
        test_infos = json.load(f)
    
    optim_levels = {}
    for i, p in enumerate(predictions):
        optim_level = test_infos[i]['optimize_level']
        if optim_level not in optim_levels:
            optim_levels[optim_level] = []
        optim_levels[optim_level].append(p)
    
    return optim_levels


def eval_all(all_predictions):
    results = {}
    for dataset, models in all_predictions.items():
        results[dataset] = {}
        for model, predictions in models.items():
            results[dataset][model] = {}
            results[dataset][model]['all'] = eval_single_results(predictions)
            
            predictions_by_optim = divide_by_optim_level(predictions)
            for optim_level, preds in predictions_by_optim.items():
                print(f"Evaluating {dataset}/{model} with optim level {optim_level}")
                eval_result = eval_single_results(preds)
                results[dataset][model][optim_level] = eval_result
                
    with open("./results.json", "w") as f:
        json.dump(results, f, indent=4)


def gen_table(rq):
    RQ_DATASETS_MAP = {"rq1": ["pseudo_code", "assembly_code"], "rq3": ["pseudo_code", "source_code_mix_pseudo_code"], "preliminary": ["source_code_mix_pseudo_code", "source_code_pseudo_code"]}
    RQ_TABLE_FILE_MAP = {"rq1": RQ1_TABLE, "rq3": RQ3_TABLE, "preliminary": PRELIMINARY_TABLE}
    
    with open(EVAL_ALL_RESULT, "r") as f:
        results = json.load(f)
    
    results = {k: v for k, v in results.items() if k in RQ_DATASETS_MAP[rq]}
    
    rows = []
    for dataset, models in results.items():
        for model, optim_levels in models.items():
            for optim_level, metrics in optim_levels.items():
                row = {
                    "model": model,
                    "dataset": dataset,
                    "optim_level": optim_level,
                    **metrics
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["model", "dataset", "optim_level"])
    df.to_csv(RQ_TABLE_FILE_MAP[rq], index=False)


def main():
    all_predictions = get_all_predictions()
    eval_all(all_predictions)
    gen_table("rq1")
    gen_table("rq3")
    gen_table("preliminary")


if __name__ == '__main__':
    main()
